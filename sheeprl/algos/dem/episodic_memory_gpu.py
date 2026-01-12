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
                 z_shape, h_shape, a_shape, 
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

        # self.solution()
        self.trajectory_length: int = trajectory_length
        self.uncertainty_threshold: float = uncertainty_threshold
        self.h_shape = h_shape      # h_shape: 4096
        self.z_shape = z_shape      # z_shape: 1024
        self.a_shape = a_shape[0]   # a_shape: (6,)
        self.key_size = self.h_shape + self.z_shape + self.a_shape
        """Size of the key vector (h, z, a)"""

        self.k_nn: int = k_nn
        self.max_elements: int = max_elements

        # self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryObject, idx, uncertainty, time of birth)

        ## CURRENTLY: if deleting 
        ## indirect connection between key & TrajObs (e.g. 3rd key = 3rd TrajObj = 3rd)
        self.trajectories_tensor: torch.Tensor = torch.empty((self.max_elements, self.key_size), device = self.device)
        ### on CPU an np because only references to Objects in RAM  (but traj data inside TrajObj on GPU)
        self.traj_obj: np.array        = np.empty(self.max_elements, dtype=object)            ## object refferences
        self.idx: torch.Tensor         = torch.empty(self.max_elements, dtype=torch.int64, device = self.device)
        self.uncertainty: torch.Tensor = torch.empty(self.max_elements, dtype=torch.float32, device = self.device)
        self.birth_time: torch.Tensor  = torch.empty(self.max_elements, dtype=torch.int64, device = self.device)
        
        self.current_trajectory: TrajectoryObject | None = None
        self.num_trajectories: int = 0  ## number of trajectories currently stored (and so also idx of last empty elem)

        self.prev_state: torch.Tensor = torch.empty(self.key_size, device = self.device)

        self.step_counter: int = 0
        # pruning stuff
        self.prune_fraction: float = prune_fraction
        self.time_to_live = time_to_live

        self.kNN_rebuild_needed: bool = True            ## if NearestNeighbors object needs to be rebuild (due to change in keys)
        self.kNN_obj: NearestNeighbors | None = None    ## NearestNeighbors object (scikit-learn oder so)
        self.key_vectors: np.array = np.empty([1])      ## array containing all keys as flatted
        self.key_array: list = []                       ## list containing all keys (this bytes shit)

        self.is_prev_stoch_state_none = True    ## used in step function to keep track of newly started episode

        warnings.warn("EpisodicMemory currently only works with a single environment instance!")

    def set_threshold(self, uncertainty_threshold: float =  0.9):
        self.uncertainty_threshold = uncertainty_threshold

    def __len__(self):
        return self.num_trajectories

    def __create_traj(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor, uncertainty: float):
        """ Create new empty trajectory, that is accessible by a key.
        """
        # key: (h_t, z_t, a_t)
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length, a_shape=self.a_shape, z_shape=self.z_shape, device=self.device) # z_shape, action_shape
        self.current_trajectory = trajectory

        # self.trajectories[key] = (trajectory, trajectory.last_idx(), uncertainty)
        # self.trajectories[key] = [trajectory, trajectory.new_traj(), uncertainty, self.step_counter]

        self.trajectories_tensor[self.num_trajectories][:self.h_shape] = h
        self.trajectories_tensor[self.num_trajectories][self.h_shape:self.h_shape+self.z_shape] = z
        self.trajectories_tensor[self.num_trajectories][-self.a_shape:] = a

        self.traj_obj[self.num_trajectories] = trajectory
        self.idx[self.num_trajectories] = trajectory.new_traj()
        self.uncertainty[self.num_trajectories] = uncertainty
        self.birth_time[self.num_trajectories] = self.step_counter

        self.num_trajectories += 1
        
    def __fill_traj(self, z: torch.Tensor, a: torch.Tensor) -> None:
        """ Adds a new transition (z, a) to the current trajectory. If the trajectory becomes full, it clears current_trajectory."""
        # value: (z_{t'}, a_{t'}) -> torch.Size([1, 1, 1024]), float (?)
        assert(self.current_trajectory is not None)
        # z_a = np.concatenate([value[0].ravel(), value[1].ravel()]) # before: z -> torch.Size([1, 1, 1024])
        if self.current_trajectory.add(z, a) == 0:
            self.current_trajectory = None


    def remove_traj(self, index: int) -> None:
        assert(index < self.num_trajectories)

        traj_obj: TrajectoryObject = self.traj_obj[index]
        traj_index = self.idx[index]
        traj_obj.del_traj(traj_index)
        
        self.trajectories_tensor[index:-1] = self.trajectories_tensor[index+1:]

        self.traj_obj[index:-1] = self.traj_obj[index+1:] 
        self.idx[index:-1] = self.idx[index+1:] 
        self.uncertainty[index:-1] = self.uncertainty[index+1:] 
        self.birth_time[index:-1] = self.birth_time[index+1:] 

        self.num_trajectories -= 1
    
    def step(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor, uncertainty: float, done:bool=False) -> None:
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
            z (torch.Tensor): The stochastic state.
            a (torch.Tensor): The action taken.
            uncertainty (float): The uncertainty of the current state.
            done (bool, optional): Whether the episode is done. Defaults to False.
        """
        # initial step, just store. Even if uncertain dont have a key for it 
        # or if deter is None while filling replay_buffer
        if self.is_prev_stoch_state_none:
            if (z is None):
                self.is_prev_stoch_state_non = True
                return
            self.prev_state[:self.h_shape] = h
            self.prev_state[self.h_shape:self.h_shape+self.z_shape] = z
            self.is_prev_stoch_state_none = False
            return
        # add new trajecory
        if uncertainty >= self.uncertainty_threshold:
            assert(self.prev_state is not None)
            # value = (z, h)          # (z_t, h_t)
            # SHAPES: h: (1, 1, 4096); z_logits: (1, 1024); real_actions: (1, 1, 1); rewards: (1,); dones: (1,)

            if self.current_trajectory is not None and self.current_trajectory.free_space != 0:
                self.__fill_traj(self.prev_state[-self.z_shape:], a)
            if len(self) >= self.max_elements:
                self._prune_memory(prune_fraction=self.prune_fraction)
            self.__create_traj(h, z, a, uncertainty)
        # just fill trajectory space
        elif self.current_trajectory is not None and self.current_trajectory.free_space != 0:
            assert(not self.is_prev_stoch_state_non)
            self.__fill_traj(self.prev_state[-self.z_shape:], a)
        # no space: no trajectory
        else:
            self.current_trajectory = None

        # manage final (done) step:
        # clear state and current trajectory as next step will be totally independend from this
        if done:
            if self.current_trajectory is not None:
                self.__fill_traj(torch.cat((z, torch.zeros((1, self.a_shape), device=self.device)), dim=1))
            self.current_trajectory = None
            self.is_prev_stoch_state_non = True
        else:
            self.prev_state[:self.h_shape] = h
            self.prev_state[self.h_shape:self.h_shape+self.z_shape] = z
            self.prev_state[-self.a_shape:] = a
            self.is_prev_stoch_state_non = False

        self.step_counter += 1

    def __str__(self):
        return f"EM| Num trajectories: {self.num_trajectories}| Trajectory length: {self.trajectory_length}| Uncertainty thr.: {self.uncertainty_threshold}| Current trajectory: {self.current_trajectory}"

    def __getitem__(self, idx):
        Exception("Not implemented yet.")
        # mem, offset, _ = self.trajectories[key]
        # sample = mem.get_trajectory(offset)
        # return sample

    def get_samples(self, skip_non_full_traj: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return all stored trajectories in batched form for training.

        Collects initial recurrent states (h), latent state sequences (z), and action
        sequences (a) from all stored trajectories and stacks them into NumPy arrays.

        Args:
            skip_non_full_traj (bool): Decides whether incomplete trajectories are also returned.
        Returns:
            tuple:
                initial_h (np.ndarray): Initial recurrent states with shape (num_trajectories, 4096).
                z_all (np.ndarray): Latent state sequences (logits) with shape (trajectory_length + 1, num_trajectories, 1024).
                a_all (np.ndarray): Action sequences with shape (trajectory_length + 1, num_trajectories, 1).
        """
        if len(self) == 0: return (None, None, None)

        ## Ones for non-invalid probability tensors
        initial_h = torch.ones((1, self.num_trajectories, self.h_shape), dtype=np.float32, device=self.device)
        z_all = torch.ones((self.trajectory_length + 1, self.num_trajectories, self.z_shape), dtype=np.float32, device=self.device)
        a_all = torch.ones((self.trajectory_length + 1, self.num_trajectories, self.a_shape), dtype=np.float32, device=self.device)
        
        full_trajes = 0
        for i in range(self.num_trajectories):
        # for i, (key, val) in enumerate(self.trajectories.items()):  # val: (TrajectoryMemory, idx, uncertainty, time of birth)
            # value
            traj_obj: TrajectoryObject = self.traj_obj[i]
            traj_nr: int = self.idx[i]
            trajectory: torch.Tensor = traj_obj.get_trajectory(traj_nr)

            # print(f"trajectory.shape[0]::{trajectory.shape[0]}      - traj_nr: {traj_nr}")
            if skip_non_full_traj and (trajectory.shape[0] != self.trajectory_length): continue
            full_trajes +=1

            z_s = torch.Tensor(trajectory[:,:-self.a_shape], device=self.device)   # ! are logits # shape (length, 1024)
            a_s = torch.Tensor(trajectory[:,-self.a_shape:], device=self.device)    # shape(length, 

            #  (h, z, a)
            z_all[0, i] = self.trajectories_tensor[i, self.h_shape:-self.a_shape]
            a_all[0, i] = self.trajectories_tensor[i, -self.a_shape:]
            # z_all[0, i] = np.frombuffer(key[1], dtype=np.float32)  # from shape (512,) into shape (1024,)
            # a_all[0, i] = np.frombuffer(key[2], dtype=np.float32) 
            z_all[1:(z_s.shape[0]+1), i] = z_s
            a_all[1:(z_s.shape[0]+1), i] = a_s
            initial_h[0, i] = self.trajectories_tensor[i, :self.h_shape]
            # initial_h[0, i] = np.frombuffer(key[0], dtype=np.float32)#.reshape(4096)
            
        # [h1, h2, ...] [zs, ...] [as, ...]
        # batch example: [h1, h2, h3]  [zs1, zs2, zs3] [as1, as2, as3]   ####### shape(sequenz, batch, 1024)
        if full_trajes == 0: return (None, None, None)
        # split empty traj parts away (from end)
        initial_h   = initial_h[:, :full_trajes]
        z_all       = z_all[:, :full_trajes]
        a_all       = a_all[:, :full_trajes]
        
        return (initial_h, z_all, a_all) 


    def _flatten_key(self, key: np.ndarray, from_bytes: bool = False):
        warnings.warn("depricated!")
        return 
        if from_bytes:
            h = np.frombuffer(key[0], dtype=np.float32).flatten()
            z = np.frombuffer(key[1], dtype=np.float32).flatten()  # from shape (512,) into shape (1024,)
            a = np.frombuffer(key[2], dtype=np.float32).flatten()
        else:
            h = key[0].flatten() ## TODO: flatten needed here??
            z = key[1].flatten()
            a = key[2].flatten()

        res = np.concatenate([h, z, a])
        return res

    def _prune_memory(self, prune_fraction: float, uncertainty_weight: float = 0.5, time_to_live_weight: float = 0.5) -> None:
        """Prune trajectories based on weighted relevancy and prune fraction.
        Args:
            prune_fraction (float): Percentage of trajectories that should be pruned.
            uncertainty_weight (float, optional): Weighting factor for uncertainty term. Defaults to 0.5.
            time_to_live_weight (float, optional): Weighting factor for time to life term. Defaults to 0.5.
        """
        to_prune_number: int = int(self.num_trajectories * prune_fraction)
        # TODO: recalc. all uncertainties or do this in extra training?

        # keys = list(self.trajectories.keys())
        # values = np.array([self.trajectories[key] for key in keys])

        # Calculate a weighted score for each trajectory
        ## TODO: uncertainty maybe too small here
        
        # High score will be deleted
        scores: torch.Tensor = uncertainty_weight * (1 - self.uncertainty) + time_to_live_weight * ((self.step_counter - self.birth_time) / self.time_to_live)
        _, to_keep_indeces = torch.topk(scores, self.num_trajectories-to_prune_number, largest=False)  ## to_prune_mask = indices

        self.trajectories_tensor = self.trajectories_tensor[to_keep_indeces]

        self.traj_obj = self.traj_obj[to_keep_indeces] 
        self.idx = self.idx[to_keep_indeces] 
        self.uncertainty = self.uncertainty[to_keep_indeces] 
        self.birth_time = self.birth_time[to_keep_indeces] 

        self.num_trajectories -= to_prune_number
        
        
    # def kNN(cloud: torch.Tensor, center: torch.Tensor, k: int = 1): # cloud: 4 dims (batch, x, y, z); center: 3 dims (x,y,z)
    #     center = center.expand(cloud.shape)
        
    #     # Computing euclidean distance
    #     dist = cloud.add( - center).pow(2).sum(dim=3).pow(.5)
        
    #     # Getting the k nearest points
    #     knn_indices = dist.topk(k, largest=False, sorted=False)[1]
        
    #     return cloud.gather(2, knn_indices.unsqueeze(-1).repeat(1,1,1,3))
    def buildKNN(self):
        """Building internal kNN object to prevent always rebuilding when multithreaded"""
        warnings.warn("depricated!")
        return 
        if self.kNN_rebuild_needed:
            self.kNN_rebuild_needed = False
            
            self.key_array = list(self.trajectories.keys())
            self.key_vectors = np.stack([
                np.concatenate([
                    np.frombuffer(k[0], dtype=np.float32),
                    np.frombuffer(k[1], dtype=np.float32),
                    np.frombuffer(k[2], dtype=np.float32),
                ])
                for k in self.key_array
            ])

            assert(len(self.key_array) > 0), "No trajectories stored in EpisodicMemory."

            search_space = np.concatenate([self._flatten_key(key_temp, from_bytes=True).reshape(1,-1) for key_temp in self.key_array])  # shape: (N, D)
            # search_space = np.concatenate([key.reshape(1,-1) for key in key_array])  # shape: (N, D)

            # x = key.reshape(1,-1)
            self.kNN_obj = NearestNeighbors(
                n_neighbors=self.k_nn,
                metric="euclidean",   # or cosine, mahalanobis, etc.
                n_jobs= -1            ## doing multithreading yes yes very very good
            ).fit(search_space)


    def kNN(self, keys: np.array, k: int = 1) -> tuple[list[tuple[np.array, np.array, np.array]], np.array]:
        """Return the k-NearestNeighbors (keys + trajectories) among stored trajectory keys.
            Args:
                keys (np.array): [1, 1024, 5126] - no bytes here object (since only called for ACD calc, not on own EM keys).
            Returns:
                neighbors_keys (np.array[tuple]): actual values, not the bytes anymore.
                trajectory_first_elems (np.array): corresponding trajectories first elems - shape: (1024, k, z_size + a_size)
        """
        warnings.warn("depricated!")
        return ([(), []])
    
        # start = time.perf_counter_ns()
        self.buildKNN()
        # print(f"BUILD KNN duration: {(time.perf_counter_ns()- start)/1000_000}ms")

        # start = time.perf_counter_ns()
        distances, indices = self.kNN_obj.kneighbors(keys) ## indices: (1024, 5)    DURATION: ~37.432707ms for 5 trajectories
        # print(f"PARALLEL KNN QUERYING duration: {(time.perf_counter_ns()- start)/1000_000}ms")

        neighbors_keys = self.key_vectors[indices]  ## (1024, 5, 5126)
        
        ## for i in tqdm.tqdm(range(neighbors_keys.shape[0] * neighbors_keys.shape[1])):

        trajectory_first_elems = np.empty((1024, 5, 1024+6))    ## (1024, 5, 1030) TODO: sizes hardcoded

        # start = time.perf_counter_ns()
        # Iterate over the indices and keys
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                key_index = int(indices[i, j])
                key = self.key_array[key_index]
                traj_obj, idx, _, _ = self.trajectories[key]
                value = traj_obj.get_trajectory(idx)
                trajectory_first_elems[i, j, :] = value[0]  # only take first (z, a) of shape (1030)
        # print(f"FOR FOR LOOPI IN KNN duration: {(time.perf_counter_ns()- start)/1000_000}ms")

        return (neighbors_keys, trajectory_first_elems)

    def kNN_gpu(self, query: torch.Tensor, k: int) -> torch.Tensor:
        """Return the k-NearestNeighbors (keys + trajectories) among stored trajectory keys.
            Args:
                query (torch.Tensor): A batch of queries.
            Returns:
                neighbors_keys (torch.Tensor): actual values, not the bytes anymore.
                trajectory_first_elems (np.array): corresponding trajectories first elems - shape: (1024, k, z_size + a_size)
        """
        metric: str = "cosine" # "cosine" | "ip"
        search_space: torch.Tensor = self.trajectories_tensor[:self.num_trajectories]   ## trajectories_tensor = torch.empty((self.max_elements, self.key_size), device = self.device)
        indices, _scores = exact_search(query, search_space, k, metric=metric) ## , device=self.device

        return indices

    def solution(self, file_path="./sheeprl/algos/dem/solution.txt"):
        try:
            with open(file_path, 'r') as file:
                solution = file.read()
                print(solution)
        except FileNotFoundError as e:
            pass
        
    def update_uncertainty(self, uncertainties: torch.Tensor) -> None:
        """Called in rehearsal training, updating uncertainty of ALL trajectories."""
        self.uncertainty = uncertainties
    
class TrajectoryObject:
    """
    Memory object for trajectories
    """
    def __init__(self, trajectory_length: int, a_shape, z_shape, device: torch.device):
        """
        Docstring for __init__
        
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        """
        self.trajectory_length: int = trajectory_length
        """The maximum length each trajectory will have."""
        self.a_shape = a_shape
        self.z_shape = z_shape
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

# class HybridKNN:
#     def __init__(self, latent_tuples, w_discrete=1.0, w_cont=1.0, include_a=True):
#         """
#         latent_tuples: list of tuples (z, h, a)
#             z: discrete latent [num_latent_dims, num_categories] (one-hot)
#             h: continuous hidden state vector
#             a: discrete action one-hot vector
#         w_discrete: weight for discrete Hamming distance
#         w_cont: weight for continuous Euclidean distance
#         include_a: whether to include action in distance
#         """
#         self.w_discrete = w_discrete
#         self.w_cont = w_cont
#         self.include_a = include_a
        
#         # Stack discrete latents and actions
#         self.Z = np.array([z.ravel() for z, _, a in latent_tuples])
#         if include_a:
#             self.A = np.array([a.ravel() for _, _, a in latent_tuples])
#         else:
#             self.A = None
        
#         # Stack continuous hidden states
#         self.H = np.array([h for _, h, _ in latent_tuples])

#     def query(self, query_tuple, k=5):
#         zq, hq, aq = query_tuple
#         zq_flat = zq.ravel()
#         hq_flat = hq.ravel()
#         if self.include_a and aq is not None:
#             aq_flat = aq.ravel()
#         else:
#             aq_flat = None
        
#         # --- Hamming distance for discrete latents ---
#         hz = np.mean(self.Z != zq_flat, axis=1)  # [num_samples]
        
#         if self.include_a and aq_flat is not None:
#             ha = np.mean(self.A != aq_flat, axis=1)
#             hamming_dist = hz + ha
#         else:
#             hamming_dist = hz
        
#         # --- Euclidean distance for continuous h ---
#         cont_dist = np.linalg.norm(self.H - hq_flat, axis=1)
        
#         # --- Hybrid distance ---
#         dist = self.w_discrete * hamming_dist + self.w_cont * cont_dist
        
#         # --- kNN ---
#         idxs = np.argsort(dist)[:k]
#         return idxs, dist[idxs]

# import torch

# class HybridKNNTorch:
#     def __init__(self, latent_tuples, device='cuda', w_discrete=1.0, w_cont=1.0, include_a=True):
#         """
#         latent_tuples: list of tuples (z, h, a)
#             z: [num_latent_dims, num_categories] one-hot
#             h: continuous hidden state vector
#             a: one-hot action
#         device: 'cuda' or 'cpu'
#         w_discrete: weight for discrete Hamming distance
#         w_cont: weight for continuous Euclidean distance
#         include_a: whether to include actions in distance
#         """
#         self.device = device
#         self.w_discrete = w_discrete
#         self.w_cont = w_cont
#         self.include_a = include_a

#         # Stack and move to device
#         self.Z = torch.stack([torch.tensor(z, dtype=torch.float32) for z, _, _ in latent_tuples]).to(device)  # [N, z_dim, num_classes]
#         if include_a:
#             self.A = torch.stack([torch.tensor(a, dtype=torch.float32) for _, _, a in latent_tuples]).to(device)
#         else:
#             self.A = None
#         self.H = torch.stack([torch.tensor(h, dtype=torch.float32) for _, h, _ in latent_tuples]).to(device)

#         # Flatten discrete tensors for distance computation
#         self.Z_flat = self.Z.flatten(start_dim=1)  # [N, z_dim*num_classes]
#         if self.include_a and self.A is not None:
#             self.A_flat = self.A.flatten(start_dim=1)  # [N, a_dim]

#     def query(self, query_tuple, k=5):
#         zq, hq, aq = query_tuple
#         zq = torch.tensor(zq, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)  # [1, z_dim*num_classes]
#         hq = torch.tensor(hq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, h_dim]
#         if self.include_a and aq is not None:
#             aq = torch.tensor(aq, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)  # [1, a_dim]

#         # --- Hamming distance for discrete parts ---
#         hz = (self.Z_flat != zq).float().mean(dim=1)  # [N]
#         if self.include_a and aq is not None:
#             ha = (self.A_flat != aq).float().mean(dim=1)
#             hamming_dist = hz + ha
#         else:
#             hamming_dist = hz

#         # --- Euclidean distance for continuous h ---
#         cont_dist = torch.norm(self.H - hq, dim=1)  # [N]

#         # --- Hybrid distance ---
#         dist = self.w_discrete * hamming_dist + self.w_cont * cont_dist

#         # --- kNN ---
#         distances, indices = torch.topk(dist, k=k, largest=False)
#         return indices.cpu().numpy(), distances.cpu().numpy()