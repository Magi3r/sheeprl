import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.neighbors import NearestNeighbors
import warnings
from lightning.fabric import Fabric
from sortedcontainers import SortedSet
import time

from sheeprl.algos.dem.knn import exact_search

class GPUEpisodicMemory():
    """
    Episodic Memory object for trajectory management running mostly on GPU.

    The EpisodicMemory class is designed to store and manage trajectories in a RL environment.
    It uses a dict to store trajectories, where each key is a tuple of (h, z, a) and the value is a tuple of (TrajectoryObject, idx, uncertainty, time of birth).
    TrajectoryObject stores overlapping trajectories inside same object.
    idx stores which i'th trajectory it is within trajectory object.
    The class provides methods for creating new trajectories, filling existing trajectories, and retrieving stored trajectories.
    """
    def __init__(self, trajectory_length: int, 
                 uncertainty_threshold: float, 
                 z_size: int, h_size: int, a_size: int, 
                 config,
                 k_nn: int = 5, 
                 max_elements: int = 1000, 
                 prune_fraction: float = 0.5,
                 time_to_live: int = 100, 
                 fabric: Fabric = None):
        """
        Docstring for __init__.

        :param trajectory_length: Length of expected trajectory.
        :type trajectory_length: int
        :param uncertainty_threshold: Threshold for uncertainty to start a new trajectory.
        :type uncertainty_threshold: float
        :param k_nn: Number of k-NearestNeighbors
        :type k_nn: int
        :param max_elements: Maximum number of trajectories stored.
        :type max_elements: int
        :param prune_fraction: Percentage of memory getting pruned.
        :type prune_fraction: float
        :param time_to_live: Time to life of trajectories (not always define the maximum, used for weighted relevancy).
        :type time_to_live: int
        """
        self.device: torch.device = fabric.device if fabric is not None else torch.device("cpu")
        self.cfg = config

        self.trajectory_length: int = trajectory_length
        self.uncertainty_threshold: float = uncertainty_threshold
        self.h_size = h_size      # h_shape: 4096
        self.z_size = z_size      # z_shape: 1024
        self.a_size = a_size      # a_shape: (6,)

        print("shapes EM: ", self.h_size, self.z_size, self.a_size)
        self.key_size = self.h_size + self.z_size + self.a_size
        """Size of the key vector (h, z, a)"""
        print("device", self.device, flush=True)

        self.k_nn: int = k_nn
        self.max_elements: int = max_elements

        # self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryObject, idx, uncertainty, time of birth)

        ## CURRENTLY: if deleting 
        ## indirect connection between key & TrajObs (e.g. 3rd key = 3rd TrajObj = 3rd)
        self.trajectories_tensor: torch.Tensor = torch.empty((self.max_elements, self.key_size), device = self.device)
        self.trajectories_tensor_knn: torch.Tensor = torch.empty((self.max_elements, self.key_size), device = self.device)
        ### on CPU an np because only references to Objects in RAM  (but traj data inside TrajObj on GPU)
        self.traj_obj: np.ndarray        = np.empty(self.max_elements, dtype=object)            ## object refferences
        self.idx: torch.Tensor         = torch.empty(self.max_elements, dtype=torch.int64, device = self.device)
        self.uncertainty: torch.Tensor = torch.empty(self.max_elements, dtype=torch.float32, device = self.device)
        self.birth_time: torch.Tensor  = torch.empty(self.max_elements, dtype=torch.int64, device = self.device)
        ## first z in each trajectory memory (so the first z_{t+1} that follows a key (h_t, z_t, a_t))
        self.next_z : torch.Tensor     = torch.empty((self.max_elements, self.z_size), dtype=torch.float32, device = self.device)
        
        self.current_trajectory: TrajectoryObject | None = None
        self.num_trajectories: int = 0  ## number of trajectories currently stored (and so also idx of first empty elem)

        self.prev_state: torch.Tensor = torch.empty(self.key_size, device = self.device)

        self.step_counter: int = 0
        # pruning stuff
        self.prune_fraction: float = prune_fraction
        self.time_to_live = time_to_live

        self.kNN_rebuild_needed: bool = True            ## if NearestNeighbors object needs to be rebuild (due to change in keys)
        self.kNN_obj: NearestNeighbors | None = None    ## NearestNeighbors object (scikit-learn oder so)
        self.key_array: list = []                       ## list containing all keys (this bytes shit) 

        self.prev_state_stored = False    ## used in step function to keep track of newly started episode

        warnings.warn("EpisodicMemory currently only works with a single environment instance!")

        self.stochastic_size = self.cfg.algo.world_model.stochastic_size
        self.discrete_size = self.cfg.algo.world_model.discrete_size

        if not config.algo.compile_em.disable:
            self.get_samples = torch.compile(self.__get_samples, fullgraph=config.algo.compile_em.fullgraph)
        else:
            self.get_samples = self.__get_samples

    def set_threshold(self, uncertainty_threshold: float =  0.9):
        self.uncertainty_threshold = uncertainty_threshold

    def __len__(self):
        return self.num_trajectories


    ##### ssshhoouuullddd be correct? 
    def __create_traj(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor, uncertainty: float):
        """ Create new empty trajectory, that is accessible by a key.
        """
        with torch.no_grad():
        # key: (h_t, z_t, a_t)
            trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length, a_size=self.a_size, z_size=self.z_size, device=self.device) # z_shape, action_shape
            self.current_trajectory = trajectory

            # self.trajectories[key] = (trajectory, trajectory.last_idx(), uncertainty)
            # self.trajectories[key] = [trajectory, trajectory.new_traj(), uncertainty, self.step_counter]

            ## shapes: torch.Size([4096]) torch.Size([1024]) torch.Size([1, 6])

            self.trajectories_tensor[self.num_trajectories][:self.h_size] = h
            self.trajectories_tensor[self.num_trajectories][self.h_size:self.h_size+self.z_size] = z
            self.trajectories_tensor[self.num_trajectories][-self.a_size:] = a

            self.trajectories_tensor_knn[self.num_trajectories][:self.h_size] = h
            if self.cfg.episodic_memory.softmax_kNN:
                self.trajectories_tensor_knn[self.num_trajectories][self.h_size:self.h_size+self.z_size] = z.view(self.stochastic_size, self.discrete_size).softmax(dim=-1).view(-1)
            else:
                self.trajectories_tensor_knn[self.num_trajectories][self.h_size:self.h_size+self.z_size] = z
            self.trajectories_tensor_knn[self.num_trajectories][-self.a_size:] = a

            if self.cfg.episodic_memory.normalize_kNN:
                self.trajectories_tensor_knn[self.num_trajectories, self.h_size:self.h_size+self.z_size] = torch.nn.functional.normalize(self.trajectories_tensor_knn[self.num_trajectories, self.h_size:self.h_size+self.z_size], p=2, dim=-1)
                self.trajectories_tensor[self.num_trajectories, :self.h_size] = torch.nn.functional.normalize(self.trajectories_tensor[self.num_trajectories, :self.h_size], p=2, dim=-1)

            self.traj_obj[self.num_trajectories] = trajectory
            self.idx[self.num_trajectories] = trajectory.new_traj()
            self.uncertainty[self.num_trajectories] = uncertainty
            self.birth_time[self.num_trajectories] = self.step_counter

            self.num_trajectories += 1
        
    ##### ssshhoouuullddd be correct? 
    def __fill_traj(self, z: torch.Tensor, a: torch.Tensor) -> None:
        """ Adds a new transition (z, a) to the current trajectory. If the trajectory becomes full, it clears current_trajectory."""
        # value: (z_{t'}, a_{t'}) -> torch.Size([1, 1, 1024]), float (?)
        assert(self.current_trajectory is not None)
        # z_a = np.concatenate([value[0].ravel(), value[1].ravel()]) # before: z -> torch.Size([1, 1, 1024])
        if self.current_trajectory.add(z, a) == 0:
            self.current_trajectory = None

    ##### ssshhoouuullddd be correct? 
    def remove_traj(self, index: int) -> None:
        assert(index < self.num_trajectories)

        traj_obj: TrajectoryObject = self.traj_obj[index]
        traj_index = self.idx[index]
        traj_obj.del_traj(traj_index)
        
        self.trajectories_tensor[index:-1] = self.trajectories_tensor[index+1:]

        self.trajectories_tensor_knn[index:-1] = self.trajectories_tensor_knn[index+1:]

        self.traj_obj[index:-1] = self.traj_obj[index+1:] 
        self.idx[index:-1] = self.idx[index+1:] 
        self.uncertainty[index:-1] = self.uncertainty[index+1:] 
        self.birth_time[index:-1] = self.birth_time[index+1:] 

        self.num_trajectories -= 1
    
    ## TODO: check is this is currect with imagined_prior, recurrent_state
    def step(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor, uncertainty: float, done:bool = False) -> None:
        """ 
        Step through the memory with new transition.

        Manages the episodic memory (build it with incoming values[state, action, uncertainty]).
        Needs to store previous state, as it will be the key if current state is uncertain.
        -----

        Start a new trajectory if uncertainty exceeds threshold.
        Fill the current trajectory with previous (state, action)
        Ends the trajectory and clears bookkeeping if done=True.

        Args:
            h (torch.Tensor): The recurrent state.
            z (torch.Tensor): The stochastic state (this was predicted based on this h + x).
            a (torch.Tensor): The action taken based on this (h, z). ~TODO: so ist das gerade zumindest
            uncertainty (float): The uncertainty of the current state.
            done (bool, optional): Whether the episode is done. Defaults to False.
        """
        # initial step, just store. Even if uncertain dont have a key for it 
        # or if deter is None while filling replay_buffer
        if not self.prev_state_stored:
            if (z is None):
                return
            self.prev_state[:self.h_size] = h
            self.prev_state[self.h_size:self.h_size+self.z_size] = z    ## ~Josch: TODO: bräuchten wir ggf. nicht nur dieses um das next_z zu setzen?
            self.prev_state[-self.a_size:] = a    ## ~Josch: TODO: bräuchten wir ggf. nicht nur dieses um das next_z zu setzen?
            self.prev_state_stored = True
            return
        # add new trajecory
        if uncertainty >= self.uncertainty_threshold:
            assert(self.prev_state is not None)
            # value = (z, h)          # (z_t, h_t)
            # SHAPES: h: (1, 1, 4096); z_logits: (1, 1, 1024); real_actions: (1, 1, 1); rewards: (1,); dones: (1,)

            # if self.current_trajectory is not None and self.current_trajectory.free_space != 0:
                # self.__fill_traj(self.prev_state[-self.z_size:], self.prev_state[-self.a_size:]) ## TODO: Josch: this should be: self.__fill_traj(z, a)????

            if len(self) >= self.max_elements:    ## TODO: -1 really necessary 
                self._prune_memory(prune_fraction=self.prune_fraction)

            self.__create_traj(self.prev_state[:self.h_size], self.prev_state[-self.z_size:], self.prev_state[-self.a_size:], uncertainty) 
            self.__fill_traj(z, a)
            ### store next z in seperate tensor for fast access during training (ACD)
            self.next_z[self.num_trajectories - 1] = z
        # just fill trajectory space
        elif self.current_trajectory is not None and self.current_trajectory.free_space != 0:
            assert(self.prev_state_stored)
            self.__fill_traj(z, a)
            # self.__fill_traj(self.prev_state[-self.z_size:], self.prev_state[-self.a_size:])
        # no space: no trajectory
        else:
            self.current_trajectory = None

        # manage final (done) step:
        # clear state and current trajectory as next step will be totally independend from this
        if done:
            if self.current_trajectory is not None:
                self.__fill_traj(z, torch.zeros_like(a, dtype=torch.float32, device=self.device))
            self.current_trajectory = None
            self.prev_state_stored = False
        else:
            self.prev_state[:self.h_size] = h
            self.prev_state[self.h_size:self.h_size+self.z_size] = z
            self.prev_state[-self.a_size:] = a
            self.prev_state_stored = True

        self.step_counter += 1

    def __str__(self):
        return f"EM| Num trajectories: {self.num_trajectories}| Trajectory length: {self.trajectory_length}| Uncertainty thr.: {self.uncertainty_threshold}| Current trajectory: {self.current_trajectory}"

    ##### ssshhoouuullddd be correct? 
    def __get_samples(self, skip_non_full_traj: bool = True) -> tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None, torch.Tensor|None]:
        """
        Return all stored trajectories in batched form for training.

        Collects initial recurrent states (h), latent state sequences (z), and action
        sequences (a) from all stored trajectories and stacks them into torch tensors (so key + traj values).

        Args:
            skip_non_full_traj (bool): Decides whether incomplete trajectories are also returned.
        Returns:
            tuple:
                initial_h (torch.tensor): Initial recurrent states with shape (1?, num_trajectories, 4096).
                z_all (torch.tensor): Latent state sequences (logits) with shape (trajectory_length + 1, num_trajectories, 1024).
                a_all (torch.tensor): Action sequences with shape (trajectory_length + 1, num_trajectories, 1).
                returned_trajs_indices (torch.tensor): Indices of trajectories actually returned (since too short trajs are skipped).
        """
        if len(self) == 0: return (None, None, None, None)

        ## Ones for non-invalid probability tensors
        initial_h   = torch.ones((1, self.num_trajectories, self.h_size), dtype=torch.float32, device=self.device)
        z_all       = torch.ones((self.trajectory_length + 1, self.num_trajectories, self.z_size), dtype=torch.float32, device=self.device)
        a_all       = torch.ones((self.trajectory_length + 1, self.num_trajectories, self.a_size), dtype=torch.float32, device=self.device)
        
        full_trajes = 0
        returned_trajs_indices = torch.empty(self.num_trajectories, dtype=torch.int64, device=self.device)

        for i in range(self.num_trajectories):
        # for i, (key, val) in enumerate(self.trajectories.items()):  # val: (TrajectoryMemory, idx, uncertainty, time of birth)
            # value
            traj_obj: TrajectoryObject = self.traj_obj[i]
            traj_nr = self.idx[i] ## torch.int64
            trajectory: torch.Tensor = traj_obj.get_trajectory(traj_nr)     ## (traj_len, z+a_size)

            # print(f"trajectory.shape[0]::{trajectory.shape[0]}      - traj_nr: {traj_nr}")
            if skip_non_full_traj and (trajectory.shape[0] != self.trajectory_length): continue
            
            returned_trajs_indices[full_trajes] = i
            full_trajes +=1

            z_s: torch.Tensor = trajectory[:,:-self.a_size].detach().clone()   # ! are logits # shape (length, 1024)
            a_s: torch.Tensor = trajectory[:,-self.a_size:].detach().clone()    # shape(length, 

            #  (h, z, a)
            z_all[0, i]     = self.trajectories_tensor[i, self.h_size:self.h_size+self.z_size]
            a_all[0, i]     = self.trajectories_tensor[i, -self.a_size:]
            initial_h[0, i] = self.trajectories_tensor[i, :self.h_size]
            # z_all[0, i] = np.frombuffer(key[1], dtype=np.float32)  # from shape (512,) into shape (1024,)
            # a_all[0, i] = np.frombuffer(key[2], dtype=np.float32) 
            z_all[1:(z_s.shape[0]+1), i] = z_s
            a_all[1:(z_s.shape[0]+1), i] = a_s
            # initial_h[0, i] = np.frombuffer(key[0], dtype=np.float32)#.reshape(4096)
            
        # [h1, h2, ...] [zs, ...] [as, ...]
        # batch example: [h1, h2, h3]  [zs1, zs2, zs3] [as1, as2, as3]   ####### shape(sequenz, batch, 1024)
        if full_trajes == 0: return (None, None, None, None)
        # split empty traj parts away (from end)
        initial_h   = initial_h[:, :full_trajes]
        z_all       = z_all[:, :full_trajes]
        a_all       = a_all[:, :full_trajes]
        
        return (initial_h, z_all, a_all, returned_trajs_indices[:full_trajes])

    def _prune_memory(self, prune_fraction: float, uncertainty_weight: float = 0.5, time_to_live_weight: float = 0.5) -> None:
        """Prune trajectories based on weighted relevancy and prune fraction.
        Args:
            prune_fraction (float): Percentage of trajectories that should be pruned.
            uncertainty_weight (float, optional): Weighting factor for uncertainty term. Defaults to 0.5.
            time_to_live_weight (float, optional): Weighting factor for time to life term. Defaults to 0.5.
        """
        print("~"*10 + " Pruning Episodic Memory " + "~"*10)

        to_prune_number: int = int(self.num_trajectories * prune_fraction)
        # TODO: recalc. all uncertainties or do this in extra training?

        # keys = list(self.trajectories.keys())
        # values = np.array([self.trajectories[key] for key in keys])

        # Calculate a weighted score for each trajectory
        ## TODO: uncertainty maybe too small here
        
        # High score will be deleted
        scores: torch.Tensor = self.uncertainty * (1 - ((self.step_counter - self.birth_time) / self.time_to_live))
        # scores: torch.Tensor = uncertainty_weight * (1 - self.uncertainty) + time_to_live_weight * ((self.step_counter - self.birth_time) / self.time_to_live)
        _, to_keep_indeces = torch.topk(scores, self.num_trajectories-to_prune_number, largest=True)  ## to_prune_mask = indices

        self.trajectories_tensor[0:self.num_trajectories-to_prune_number] = self.trajectories_tensor[to_keep_indeces]
        self.trajectories_tensor_knn[0:self.num_trajectories-to_prune_number] = self.trajectories_tensor_knn[to_keep_indeces]

        self.traj_obj[0:self.num_trajectories-to_prune_number] = self.traj_obj[to_keep_indeces.cpu()]
        self.idx[0:self.num_trajectories-to_prune_number] = self.idx[to_keep_indeces] 
        self.uncertainty[0:self.num_trajectories-to_prune_number] = self.uncertainty[to_keep_indeces] 
        self.birth_time[0:self.num_trajectories-to_prune_number] = self.birth_time[to_keep_indeces] 

        self.num_trajectories -= to_prune_number
        
    def kNN_gpu(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the k-NearestNeighbors (keys + trajectories) among stored trajectory keys.
            Args:
                query (torch.Tensor): A batch of queries.
            Returns:
                indices (torch.Tensor): indices of the k nearest neighbors for each query.
                scores (torch.Tensor): similarity scores of the k nearest neighbors for each query.
        """
        with torch.no_grad():
            metric: str = "cosine" # "cosine" | "ip"

            if self.cfg.episodic_memory.softmax_kNN:
                query[:, self.h_size:self.h_size+self.z_size] = query[:, self.h_size:self.h_size+self.z_size].view(query.shape[0], self.stochastic_size, self.discrete_size).softmax(dim=-1).view(query.shape[0], -1)

            if self.cfg.episodic_memory.normalize_kNN:
                query[:, self.h_size:self.h_size+self.z_size] = torch.nn.functional.normalize(query[:, self.h_size:self.h_size+self.z_size], p=2, dim=-1)
                query[:, :self.h_size]                        = torch.nn.functional.normalize(query[:, :self.h_size], p=2, dim=-1)
                
            search_space: torch.Tensor = self.trajectories_tensor_knn[:self.num_trajectories]   ## trajectories_tensor = torch.empty((self.max_elements, self.key_size), device = self.device)
            indices, scores = exact_search(query, search_space, k, metric=metric) ## , device=self.device

            return indices, scores

    def solution(self, file_path="./sheeprl/algos/dem/solution.txt"):
        try:
            with open(file_path, 'r') as file:
                solution = file.read()
                print(solution)
        except FileNotFoundError as e:
            pass
    
class TrajectoryObject:
    """
    Memory object for trajectories
    """
    def __init__(self, trajectory_length: int, a_size: int, z_size: int, device: torch.device):
        """
        Docstring for __init__
        
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        """
        self.trajectory_length: int = trajectory_length
        """The maximum length each trajectory will have."""
        self.a_shape = a_size
        self.z_shape = z_size
        self.memory_width = self.z_shape+self.a_shape
        """How many entries each row will have."""
        
        self.traj_num_to_offset_size_increase = 10
        """How many trajectories we expect to be in the Object. Optimal value depends on multiple factors."""
        
        self.free_space: int = trajectory_length
        """How much free space this object still has. Always resets if new trajectory starts."""
        # self.memory: np.array = np.ones((trajectory_length, self.memory_width), dtype=np.float32)  # TODO: add size of tuple (z_t', a_t') ~ 1024+6 TODO: add to device when using tensors?
        self.memory: torch.Tensor = torch.ones((trajectory_length, self.memory_width), dtype=torch.float32, device=device)
        """"The actual trajectories."""

        # self.traj_num_to_offset : np.array = np.zeros((self.traj_num_to_offset_size_increase,), dtype=int) # 10 is test value for now
        self.traj_num_mapping: Map = Map()
        """"The actual trajectory starting index."""
        self.num_trajectories : int = 0
        """"Trajectory counter."""
        
        self.device: torch.device = device

    def new_traj(self):
        """"Extend the internal memory, so it can hold another trajectory.

        :return: The internal number for the new trajectory in this object.
        """
        nr_idx = self.num_trajectories
        self.num_trajectories += 1

        # if self.traj_num_to_offset.shape[0] <= self.num_trajectories + 1:
        #     self.traj_num_to_offset = np.concatenate(
        #         (self.traj_num_to_offset, np.zeros((self.traj_num_to_offset_size_increase,), dtype=int)),
        #         axis=0
        #     )

        # possible if lenght-freespace = 0 ??? # TODO: add size of tuple (z_t', a_t')
        self.memory = torch.concatenate((self.memory, torch.ones((self.trajectory_length-self.free_space, self.memory_width), dtype=torch.float32, device=self.device)), axis=0)
        self.free_space = self.trajectory_length
        # self.traj_num_to_offset[nr_idx] = self.last_idx()
        
        self.traj_num_mapping.append(self.last_idx())

        return nr_idx

    def del_traj(self, traj_nr: int) -> None:
        """ This function deletes a trajectory (a contiguous block of entries) from the internal memory based on a given trajectory number.
        It: Computes the start and end indices of the trajectory to remove using traj_num_to_offset.
        Removes that slice from self.memory.
        Shifts all subsequent data left to fill the gap.
        Updates traj_num_to_offset so that indices of following trajectories are decremented by the length of the deleted trajectory.
        So 'traj_nr' will now point to the start of the 'traj_nr' + 1 trajectory
        Handles edge cases such as deleting the last trajectory or an empty trajectory.
        Delete a trajectory by its number 

        Args:
            traj_nr (int): internal trajectory number to delete.
        """
        prev = self.traj_num_mapping.get_prev_index(traj_nr)
        start_idx = 0 if prev == -1 else self.traj_num_mapping[prev] + self.trajectory_length

        next_ = self.traj_num_mapping.get_next_index(traj_nr)
        end_idx = self.memory.shape[0] if next_ == -1 else self.traj_num_mapping[next_] 

        to_delete = (end_idx - start_idx)
        
        if to_delete > 0:
            ## np.concatenate([array([], dtype=int64), array([2, 3, 4])], axis=0) -> array([2, 3, 4]) ~Good
            # self.memory = torch.concatenate([self.memory[:start_idx], self.memory[end_idx:]], axis = 0) # +1-1*1/1????
            ### safe good :)):
            self.memory[start_idx:end_idx] = self.memory[start_idx+end_idx:]
            self.memory = self.memory[:start_idx+end_idx]
            # decrement following trajectory indices by length of deleted trajectory, if not last element
            if next_ != -1:
                self.traj_num_mapping.add_to_all_following(traj_nr, -(end_idx - start_idx))
            else:
                self.traj_num_mapping.delete(traj_nr)

                self.free_space = max(0, self.last_idx() - start_idx)
            self.kNN_rebuild_needed = True

        elif to_delete == 0:
            pass

        else:
            self.traj_num_mapping.delete(traj_nr)
            self.kNN_rebuild_needed = True
            
    def add(self, z: torch.Tensor, a: torch.Tensor) -> int:
        """ Add a value into the trajectory.
                
        Args:
            value: The value

        Returns: 
            free_space (int): The remaining free space.
        """
        self.kNN_rebuild_needed = True
        self.memory[-self.free_space][:self.z_shape] = z
        self.memory[-self.free_space][-self.a_shape:] = a
        self.free_space -= 1
        return self.free_space

    def last_idx(self):
        """Get index of last stored element in memory (including or excluding?)"""
        return self.memory.shape[0] - self.free_space

    def __str__(self):
        return f"TrajectoryObj| Free space: {self.free_space}| Trajectory length: {self.trajectory_length}"
    
    def get_trajectory(self, traj_nr) -> torch.Tensor:
        """Returns a specific trajectory in full"""
        start_idx = self.traj_num_mapping[traj_nr]
        end_idx = min(self.last_idx()+1, start_idx + self.trajectory_length)
        # print("get_traj: ", start_idx, end_idx, self.last_idx(), start_idx + self.trajectory_length)
        return self.memory[start_idx:end_idx]

class Map():
    """ Is doing stuff.
    Sehr coole Klasse
    """
    def __init__(self):
        self.sorted_indices = SortedSet()
        self.size_increase: int = 10
        self.map: np.array = np.zeros(self.size_increase, dtype=int)
        self.i: int = 0

    def append(self, value):
        index = self.i
        self.sorted_indices.add(index)
        self.map[index] = value
        self.i+=1
        if self.i == self.map.shape[0]:
            self.map = np.concatenate((self.map, np.zeros(self.size_increase, dtype=int)), axis=0) # TypeError: only integer scalar arrays can be converted to a scalar index
        
    def get_prev_index(self, index) -> int:
        idx = self.sorted_indices.bisect_left(index)
        if idx > 0:
            return self.sorted_indices[idx - 1]
        return -1

    def get_next_index(self, index) -> int:
        idx = self.sorted_indices.bisect_right(index)
        if idx < len(self.sorted_indices):
            return self.sorted_indices[idx]
        return -1
    
    def delete(self, index):
        self.sorted_indices.discard(index)
    
    def add_to_all_following(self, index, amount):
        temp = np.zeros_like(self.map)
        temp[index+1:] = amount

        self.map += temp

    def __contains__(self, index):
        return index in self.sorted_indices

    def __getitem__(self, index):
        return self.map[index]
    
    def __setitem__(self, index, value):
        if index >= self.i:
            raise IndexError
        
        self.map[index] = value

    def __delitem__(self, index):
        self.delete(index)