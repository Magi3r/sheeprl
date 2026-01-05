import numpy as np
import hnswlib
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.neighbors import KNeighborsClassifier
import warnings
from lightning.fabric import Fabric
from sortedcontainers import SortedSet

class EpisodicMemory():
    """
    Episodic Memory object for trajectory management.

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
        # self.solution()
        self.trajectory_length: int = trajectory_length
        self.uncertainty_threshold: float = uncertainty_threshold
        self.key_size = h_shape # + z_shape + a_shape # TODO: will probably need to change if shapes are not 1D
        """Size of the key vector (h, z, a)"""
        self.a_shape = a_shape[0]   # a_shape: (6,)
        self.h_shape = h_shape      # h_shape: 4096
        self.z_shape = z_shape      # z_shape: 1024

        self.k_nn: int = k_nn

        self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryObject, idx, uncertainty, time of birth)

        self.current_trajectory: TrajectoryObject | None = None

        self.prev_state = None

        self.hnsw_storage: hnswlib.Index = None
        self._init_hnsw(max_elements=max_elements*2, dim = 32*32+512+6) # TODO: insert shapes properly hier error?
        self.step_counter = 0
        self.max_elements: int = max_elements
        # pruning stuff
        self.prune_fraction: int = prune_fraction
        self.time_to_live = time_to_live

        self.device: torch.device = fabric.device if fabric is not None else torch.device("cpu")

        # print(f"init EM with following shapes: z:{z_shape}, h:{h_shape}, a:{a_shape}, key_size: {self.key_size}")
        warnings.warn("EpisodicMemory currently only works with a single environment instance!")

    def __len__(self):
        return len(self.trajectories)

    def __create_traj(self, key: tuple, uncertainty: float):
        """ Create new empty trajectory, that is accessible by a key.
        """
        # key: (h_t, z_t, a_t)
        # uncertainty: float
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length, a_shape=self.a_shape, z_shape=self.z_shape, device=self.device) # z_shape, action_shape
        self.current_trajectory = trajectory

        # self.trajectories[key] = (trajectory, trajectory.last_idx(), uncertainty)
        self.trajectories[key] = [trajectory, trajectory.new_traj(), uncertainty, self.step_counter]
        
    def __fill_traj(self, value: tuple):
        """ Adds a new transition (z, a) to the current trajectory. If the trajectory becomes full, it clears current_trajectory."""
        # value: (z_{t'}, a_{t'}) -> torch.Size([1, 1, 1024]), float (?)
        assert(self.current_trajectory is not None)
        z_a = np.concatenate([value[0].ravel(), value[1].ravel()]) # before: z -> torch.Size([1, 1, 1024])
        if self.current_trajectory.add(z_a) == 0:
            self.current_trajectory = None

    # !! use .tobytes() for key elements
    def remove_traj(self, key: tuple):
        """ Remove trajectory by its key """
        assert(key in self.trajectories)
        traj_obj, idx, _, _ = self.trajectories[key]
        traj_obj.del_traj(idx)
        del self.trajectories[key]

    def step(self, state: dict, action: np.ndarray, uncertainty: float, done:bool=False):
        """ 
        Step through the memory with new transition.

        Manages the episodic memory (build it with incoming values[state, action, uncertainty]).
        Needs to store previous state, as it will be the key if current state is uncertain.
        -----

        Start a new trajectory if uncertainty exceeds threshold.
        Fill the current trajectory with previous (state, action)
        Ends the trajectory and clears bookkeeping if done=True.

        Args:
            state (dict): The current state, containing 'deter' and 'stoch' keys.
            action (np.ndarray): The action taken in the current state.
            uncertainty (float): The uncertainty of the current state.
            done (bool, optional): Whether the episode is done. Defaults to False.
        """
        # initial step, just store. Even if uncertain dont have a key for it 
        # or if deter is None while filling replay_buffer
        if self.prev_state == None or self.prev_state["deter"] is None:
            self.prev_state = state
            return
        action = action.astype(np.float32)
        # add new trajecory
        if uncertainty > self.uncertainty_threshold:
            assert(self.prev_state is not None)
            # value = (z, h)          # (z_t, h_t)
            # print("asdfjklössssss ", action, action.tobytes()) # asdfjklössssss  [[[5]]] b'\x05\x00\x00\x00\x00\x00\x00\x00'
            key = (self.prev_state["deter"].tobytes(), self.prev_state["stoch"].tobytes(), action.tobytes())    # (h_t, z_t, a_t) # for now fatten so dict can hash it

            # print("self.prev_state['stoch'].tobytes() len:", len(self.prev_state["stoch"].tobytes()))
            # print("self.prev_state['deter'].tobytes() len:", len(self.prev_state["deter"].tobytes()))
            
            # SHAPES:
            #   h: (1, 1, 4096)
            #   z_logits: (1, 1024)
            #   real_actions: (1, 1, 1)
            #   rewards:      (1,)
            #   dones:        (1,)
            # EM| Num trajectories: 0| Trajectory length: 20| Uncertainty thr.: 0.9| Current trajectory: None
            # self.prev_state['stoch'].tobytes() len: 4096
            # self.prev_state['deter'].tobytes() len: 16384 = 4bytes * 4096

            if self.current_trajectory is not None and self.current_trajectory.free_space != 0:
                self.__fill_traj((self.prev_state["stoch"], action))
            if len(self) >= self.max_elements:
                # print("~~prune due to full EM!~~")
                self._prune_memory(prune_fraction=self.prune_fraction)
            self.__create_traj(key, uncertainty)
        # just fill trajectory space
        elif self.current_trajectory is not None and self.current_trajectory.free_space != 0:
            assert(self.prev_state is not None)
            
            self.__fill_traj((self.prev_state["stoch"], action))
        # no space: no trajectory
        else:
            self.current_trajectory = None

        # manage final (done) step:
        # clear state and current trajectory as next step will be totally independend from this
        if done:
            if self.current_trajectory is not None:
                self.__fill_traj((state["stoch"], np.zeros_like(action).astype(np.float32)))
            self.current_trajectory = None
            self.prev_state = None
        else:
            self.prev_state = state #.copy()?

        self.step_counter += 1

    # def _flatten_key(self, key):
    #     """Convert (z, h, a) tensors into one numpy vector."""
    #     z, h, a = key  # each is a torch tensor
    #     z = z.flatten().cpu().numpy()
    #     h = h.flatten().cpu().numpy()
    #     a = a.flatten().cpu().numpy()
    #     return np.concatenate([z, h, a], axis=0)
    
    def __str__(self):
        return f"EM| Num trajectories: {len(self.trajectories)}| Trajectory length: {self.trajectory_length}| Uncertainty thr.: {self.uncertainty_threshold}| Current trajectory: {self.current_trajectory}"


    def __getitem__(self, idx):
        Exception("Not implemented yet.")
        # mem, offset, _ = self.trajectories[key]
        # sample = mem.get_trajectory(offset)
        # return sample

    def get_samples(self, skip_non_full_traj: bool = True) -> tuple[np.array, np.array, np.array]:
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
        # TODO: hardcoded shapes!!!
        if len(self) == 0: return (None, None, None)

        ## Ones for non-invalid probability tensors
        initial_h = np.ones((1, len(self.trajectories), self.h_shape), dtype=np.float32)
        z_all = np.ones((self.trajectory_length + 1, len(self.trajectories), self.z_shape), dtype=np.float32)
        a_all = np.ones((self.trajectory_length + 1, len(self.trajectories), self.a_shape), dtype=np.float32)
        
        full_trajes = 0
        for i, (key, val) in enumerate(self.trajectories.items()):  # val: (TrajectoryMemory, idx, uncertainty, time of birth)
            # value
            traj_obj: TrajectoryObject = val[0]
            traj_nr: int = val[1]
            trajectory: np.array = traj_obj.get_trajectory(traj_nr)
            

            # single rehearsal train took: 1.9778013229370117e-05 seconds ~EM length: 5
            # training per_rank_gradient_steps:1, seq_len: 64 times took 0.0007024538516998291 seconds 
            # trajectory.shape[0]::5
            # trajectory.shape[0]::1
            # trajectory.shape[0]::0
            # trajectory.shape[0]::0
            # trajectory.shape[0]::0
            # trajectory.shape[0]::0
            # batch_size: 1; i:0; j:1

            # print(f"trajectory.shape[0]::{trajectory.shape[0]}      - traj_nr: {traj_nr}")
            if skip_non_full_traj and (trajectory.shape[0] != self.trajectory_length): continue
            full_trajes +=1

            z_s = np.array(trajectory[:,:-self.a_shape])   # ! are logits # shape (length, 1024)
            a_s = np.array(trajectory[:,-self.a_shape:])    # shape(length, 

            z_all[0, i] = np.frombuffer(key[1], dtype=np.float32)  # from shape (512,) into shape (1024,)
            a_all[0, i] = np.frombuffer(key[2], dtype=np.float32) 
            z_all[1:(z_s.shape[0]+1), i] = z_s
            a_all[1:(z_s.shape[0]+1), i] = a_s
            initial_h[0, i] = np.frombuffer(key[0], dtype=np.float32)#.reshape(4096)
            
        # [h1, h2, ...] [zs, ...] [as, ...]
        # batch example: [h1, h2, h3]  [zs1, zs2, zs3] [as1, as2, as3]   ####### shape(sequenz, batch, 1024)
        if full_trajes == 0: return (None, None, None)
        # split empty traj parts away (from end)
        initial_h   = initial_h[:, :full_trajes]
        z_all       = z_all[:, :full_trajes]
        a_all       = a_all[:, :full_trajes]
        
        return (initial_h, z_all, a_all) 

    # def add(self, key: tuple, value: tuple, uncertainty: float):
    #     """ Add new trajectory """
    #     # key: (h_t, z_t, a_t)
    #     # value: (z_{t'}, a_{t'})
    #     # uncertainty: float
    #     trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length) # z_shape, action_shape

    #     trajectory.add(value)

    def _flatten_key(self, k: torch.Tensor):
        z, h, a = k  # each is a torch tensor
        z = z.detach().cpu().numpy().flatten()
        h = h.detach().cpu().numpy().flatten()
        a = a.detach().cpu().numpy().flatten()

        res = np.concatenate([z, h, a])
        return res

    def _init_hnsw(self, max_elements, dim, rebuild=False):
        # M: max outgoing connections in graph, higher is better but slower, highly connected to dims
        # ef_construction: construction accuracy/speed tradeoff
        self.hnsw_storage = hnswlib.Index(space="l2", dim=dim)
        self.hnsw_storage.init_index(max_elements, M = 16, ef_construction = 100, allow_replace_deleted=True)
        # ef: query accuracy
        self.hnsw_storage.set_ef(50)
        if rebuild:
            assert(max_elements>=len(self.trajectories))
            keys = np.array(list(map(lambda x: self._flatten_key(x).reshape(1, -1), self.trajectories.keys())))
            self.hnsw_storage.add_items(keys)

    def _prune_memory(self, prune_fraction: float, uncertainty_weight: float = 0.5, time_to_live_weight: float = 0.5) -> None:
        """Prune trajectories based on weighted relevancy and prune fraction.
        Args:
            prune_fraction (float): Percentage of trajectories that should be pruned.
            uncertainty_weight (float, optional): Weighting factor for uncertainty term. Defaults to 0.5.
            time_to_live_weight (float, optional): Weighting factor for time to life term. Defaults to 0.5.
        """
        deleted_trajs: int = 0

        to_prune: int = int(len(self.trajectories) * prune_fraction)
        # uncertainty_threshold: float = self.uncertainty_threshold

        # TODO: recalc. all uncertainties or do this in extra training?

        keys = list(self.trajectories.keys())
        values = np.array([self.trajectories[key] for key in keys])

        # Calculate a weighted score for each trajectory
        ## TODO: uncertainty too small here
        scores = uncertainty_weight * (1 - values[:, 2]) + time_to_live_weight * (values[:, 3] / self.time_to_live)

        # Get the indices of the trajectories with the lowest scores
        indices_to_prune = np.argsort(scores)[-to_prune:]

        # Prune the selected trajectories
        # print("argsort(scores)  :", np.argsort(scores))
        # print("scores           :", scores)
        for i in indices_to_prune:
            # print(f"prune {scores[i]}") # maybe this print does not make so much sense??
            self.remove_traj(keys[i])
            deleted_trajs += 1

    def _add_hnsw(self, key):
        if self.hnsw_storage.get_current_count() == self.hnsw_storage.get_max_elements():
            print("Problem")
        key = np.array([self._flatten_key(key).reshape(1, -1)])
        self.hnsw_storage.add_items(key)
        
    # def kNN(cloud: torch.Tensor, center: torch.Tensor, k: int = 1): # cloud: 4 dims (batch, x, y, z); center: 3 dims (x,y,z)
    #     center = center.expand(cloud.shape)
        
    #     # Computing euclidean distance
    #     dist = cloud.add( - center).pow(2).sum(dim=3).pow(.5)
        
    #     # Getting the k nearest points
    #     knn_indices = dist.topk(k, largest=False, sorted=False)[1]
        
    #     return cloud.gather(2, knn_indices.unsqueeze(-1).repeat(1,1,1,3))
        
    def kNN(self, key: tuple, k: int = 1) -> tuple[tuple, list[None, None, None, None]]:
        """Return the k-nearest neighbors (keys + trajectories) among stored trajectory keys."""
        
        def flatten_key(k):
            z, h, a = k  # each is a torch tensor
            z = z.detach().cpu().numpy().flatten()
            h = h.detach().cpu().numpy().flatten()
            a = a.detach().cpu().numpy().flatten()

            res = np.concatenate([z, h, a])
            return res
        
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        key_array = list(self.trajectories.keys())
        assert(len(key_array) > 0), "No trajectories stored in EpisodicMemory."
        search_space = np.stack([flatten_key(key) for key in key_array])  # shape: (N, D)

        x = flatten_key(key).reshape(1, -1)
        knn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"  # or cosine, mahalanobis, etc.
        ).fit(search_space)

        distances, indices = knn.kneighbors(x) # [None]
        
        # Return the actual trajectory objects
        keys = list(self.trajectories.keys())
        neighbors_keys = [keys[i] for i in indices[0]]
        neighbors_trajecories = [self.trajectories[keys[i]] for i in indices[0]]

        return (neighbors_keys, neighbors_trajecories)

    def solution(self, file_path="./sheeprl/algos/dem/solution.txt"):
        try:
            with open(file_path, 'r') as file:
                solution = file.read()
                print(solution)
        except FileNotFoundError as e:
            pass
        
    def update_uncertainty(self, key: tuple[None, None, None], uncertainty: float) -> None:
        obj, idx, _, ttl = self.trajectories[key]
        self.trajectories[key] = (obj, idx, uncertainty, ttl)

    # def kNN(self, key, k:int=1):
    #     # key: (h_t, z_t, a_t)
    #     Warning("For now ignore z_t.")

    #     # (h_t, z_t, a_t) -> (h_t, z_t)
    #     # entries = np.array([[k[0], k[1]] for k in self.trajectories.keys()])


    #     # return k nearest neighbors based on some distance metric
    #     raise NotImplementedError("kNN method not implemented yet.")
    
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
        self.memory: np.array = np.ones((trajectory_length, self.memory_width), dtype=np.float32)  # TODO: add size of tuple (z_t', a_t') ~ 1024+6 TODO: add to device when using tensors?
        """"The actual trajectories."""

        self.traj_num_to_offset : np.array = np.zeros((self.traj_num_to_offset_size_increase,), dtype=int) # 10 is test value for now
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

        if self.traj_num_to_offset.shape[0] <= self.num_trajectories + 1:
            self.traj_num_to_offset = np.concatenate(
                (self.traj_num_to_offset, np.zeros((self.traj_num_to_offset_size_increase,), dtype=int)),
                axis=0
            )

        self.memory = np.concatenate((self.memory, np.ones((self.trajectory_length-self.free_space, self.memory_width), dtype=np.float32)), axis=0) # possible if lenght-freespace = 0 ??? # TODO: add size of tuple (z_t', a_t')
        self.free_space = self.trajectory_length
        self.traj_num_to_offset[nr_idx] = self.last_idx()
        
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
        # start_idx = self.traj_num_to_offset[traj_nr-1]+self.trajectory_length if traj_nr-1 >=0 else 0
        # end_idx = self.traj_num_to_offset[traj_nr+1] if traj_nr < self.traj_num_to_offset.shape[0]-1 else self.memory.shape[0]
        
        prev = self.traj_num_mapping.get_prev_index(traj_nr)
        start_idx = 0 if prev == -1 else self.traj_num_mapping[prev] + self.trajectory_length

        next_ = self.traj_num_mapping.get_next_index(traj_nr)
        end_idx = self.memory.shape[0] if next_ == -1 else self.traj_num_mapping[next_] 

        # [0,3,5,10,12]
        # del 2
        # [0,3,5,5,7]
        # [0,3,5,10,12]
        # print("DELETE TRAJ NR:", traj_nr, " FROM ", start_idx, " TO ", end_idx)

        to_delete = (end_idx - start_idx)
        # 8-3 = 5
        # 0 to 10 = 11 elements
        # 0to 2 + 8to 10 = 3 + 6 elements
        
        if to_delete > 0:
            # if self.memory.shape[0] != end_idx:
            ## np.concatenate([array([], dtype=int64), array([2, 3, 4])], axis=0) -> array([2, 3, 4]) ~Good
            self.memory = np.concatenate([self.memory[:start_idx], self.memory[end_idx:]], axis = 0) # +1-1*1/1????
            # elif start_idx == 0:
            #     self.memory = self.memory[end_idx:]
            # else:
            #     self.memory = self.memory[:start_idx]

            # decrement following trajectory indices by length of deleted trajectory, if not last element
            if next_ != -1:
                self.traj_num_mapping.add_to_all_following(traj_nr, -(end_idx - start_idx))
                # temp = np.zeros_like(self.traj_num_to_offset)
                # temp[traj_nr+1:] = (end_idx - start_idx)

                # self.traj_num_to_offset -= temp
            # traj_nr marks the last trajectory
            else:
                if prev != -1:
                    new_length = self.traj_num_mapping[prev] + self.trajectory_length
                else:
                    new_length = 1
                self.traj_num_mapping.delete(traj_nr)
                        
                self.free_space = max(0, self.last_idx() - start_idx)                
        
        elif to_delete == 0:
            pass

        else:
            self.traj_num_mapping.delete(traj_nr)
            
    def add(self, value: np.array) -> int:
        """ Add a value into the trajectory.
                
        Args:
            value: The value

        Returns: 
            free_space (int): The remaining free space.
        """
        self.memory[-self.free_space] = value # TODO: add tuple (z_t', a_t') |IndexError: index -5 is out of bounds for axis 0 with size 1
        self.free_space -= 1
        # print("ADD VALUE:", value)
        # print("memory: ", type(self.memory[0]))
        # print(type(self.memory))

        # ADD VALUE: (array([[[0., 0., 0., ..., 0., 0., 0.]]], dtype=float32), array([[2]]))
        # memory:  <class 'numpy.ndarray'>
        # <class 'numpy.ndarray'>
        return self.free_space

    def last_idx(self):
        """Get index of last stored element in memory (including or excluding?)"""
        return self.memory.shape[0] - self.free_space

    def __str__(self):
        return f"TrajectoryObj| Free space: {self.free_space}| Trajectory length: {self.trajectory_length} \
            | Traj num. to offset: {self.traj_num_to_offset}"
    
    def get_trajectory(self, traj_nr) -> np.array:
        """Returns a specific trajectory in full"""
        start_idx = self.traj_num_to_offset[traj_nr]
        end_idx = min(self.last_idx()+1, start_idx + self.trajectory_length)
        # print("get_traj: ", start_idx, end_idx, self.last_idx(), start_idx + self.trajectory_length)
        return self.memory[start_idx:end_idx]
    
    # depricated
    # def get_all_trajectories(self) -> list[np.array]:
    #     """Return a list of all trajecotries, without indices"""
    #     trajectories = []

    #     for traj_nr in range(self.num_trajectories):
    #         start_idx = self.traj_num_to_offset[traj_nr-1]+self.trajectory_length if traj_nr-1 >=0 else 0
    #         end_idx = start_idx + self.trajectory_length

    #         trajectories.append(np.array(self.memory[start_idx:end_idx]))
        
    #     return trajectories

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
        
    def get_prev_index(self, index) -> None:
        idx = self.sorted_indices.bisect_left(index)
        if idx > 0:
            return self.sorted_indices[idx - 1]
        return -1

    def get_next_index(self, index) -> None:
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