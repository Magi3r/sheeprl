import numpy as np
import hnswlib
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.neighbors import KNeighborsClassifier
import warnings

class EpisodicMemory():
    """
    Episodic Memory object for trajectory management.

    general idea: store overlapping trajectories inside same object.
                  keep dictionary of (z,h,a): (trajectoryobject, nr, uncertainty)
                    nr stores the number within trajectory object
                ...
    """
    def __init__(self, trajectory_length: int, uncertainty_threshold: float, z_shape, h_shape, a_shape, k_nn: int = 5, max_elements: int = 1000):
        """
        Docstring for __init__
        
        :param self: Description
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        :param uncertainty_threshold: Threshold for uncertainty to start a new trajectory
        :type uncertainty_threshold: float
        """
        self.trajectory_length: int = trajectory_length
        self.uncertainty_threshold: float = uncertainty_threshold
        """Threshold for uncertainty to start a new trajectory"""
        self.key_size = h_shape # + z_shape + a_shape # TODO: will probably need to change if shapes are not 1D
        """Size of the key vector (h, z, a)"""
        self.a_shape = a_shape
        self.h_shape = h_shape
        self.z_shape = z_shape

        self.k_nn: int = k_nn
        """Number of nearest neighbors to retrieve."""

        self.trajectories: dict = {} # key: (h_t, z_t, a_t), value: (TrajectoryMemory, idx, uncertainty, time of birth)

        self.current_trajectory: TrajectoryObject | None = None

        self.prev_state = None

        self.hnsw_storage: hnswlib.Index = None
        self._init_hnsw(max_elements=max_elements*2, dim = 32*32+512+6) # TODO: insert shapes properly hier error?
        self.step_counter = 0
        self.max_elements = max_elements

        print(f"init EM with following shapes: z:{z_shape}, h:{h_shape}, a:{a_shape}, key_size: {self.key_size}")
        warnings.warn("EpisodicMemory currently only works with a single environment instance!")

    def __len__(self):
        return len(self.trajectories)

    def __create_traj(self, key: tuple, uncertainty: float):
        """ Create new trajectory 
            Create an empty trajectory, that is accessible by a key 
        """
        # key: (h_t, z_t, a_t)
        # uncertainty: float
        trajectory = self.current_trajectory if self.current_trajectory else TrajectoryObject(self.trajectory_length) # z_shape, action_shape
        self.current_trajectory = trajectory

#   File "/home/dude/Desktop/sheeprl/sheeprl/algos/dem/episodic_memory.py", line 63, in __create_traj
#     self.trajectories[key] = [trajectory, trajectory.new_traj(), uncertainty, self.step_counter]
# TypeError: unhashable type: 'numpy.ndarray'

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
        traj_obj, idx, _ = self.trajectories[key]
        traj_obj.del_traj(idx)
        del self.trajectories[key]

    def step(self, state: dict, action: np.ndarray, uncertainty: float, done:bool=False):
        """ Step through the memory with new transition 
        called each step :)

        Manages the episodic memory (build it with incoming values[state, action, uncertainty]).
        needs to store previous state, as it will be the key if current state is uncertain.
        -----

        Start a new trajectory if uncertainty exceeds threshold.
        Fill the current trajectory with previous (state, action)
        Ends the trajectory and clears bookkeeping if done=True.

        acion: np.ndarray containing numpy.dtypes.Int64DType
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
            print("asdfjklössssss ", action, action.tobytes()) # asdfjklössssss  [[[5]]] b'\x05\x00\x00\x00\x00\x00\x00\x00'
            key = (self.prev_state["deter"].tobytes(), self.prev_state["stoch"].tobytes(), action.tobytes())    # (h_t, z_t, a_t) # for now fatten so dict can hash it

            print("self.prev_state['stoch'].tobytes() len:", len(self.prev_state["stoch"].tobytes()))
            print("self.prev_state['deter'].tobytes() len:", len(self.prev_state["deter"].tobytes()))
            
            # SHAPES:
            #   h: (1, 1, 4096)
            #   z_logits: (1, 1024)
            #   real_actions: (1, 1, 1)
            #   rewards:      (1,)
            #   dones:        (1,)
            # EM| Num trajectories: 0| Trajectory length: 20| Uncertainty thr.: 0.9| Current trajectory: None
            # self.prev_state['stoch'].tobytes() len: 4096
            # self.prev_state['deter'].tobytes() len: 16384 = 4bytes * 4096

            if self.current_trajectory is not None:
                self.__fill_traj((self.prev_state["stoch"], action))
            self.__create_traj(key, uncertainty)
        # just fill trajectory space
        elif self.current_trajectory is not None:
            assert(self.prev_state is not None)
            
            self.__fill_traj((self.prev_state["stoch"], action))
        # no space: no trajectory
        else:
            self.current_trajectory = None

        # manage final (done) step:
        # clear state and current trajectory as next step will be totally independend from this
        if done:
            if self.current_trajectory is not None:
                self.__fill_traj((state["stoch"], None))
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

    def get_samples(self) -> tuple[np.array, np.array, np.array]:
        """
        Return all stored trajectories in batched form for training.

        Collects initial recurrent states, latent state sequences (z), and action
        sequences (a) from all stored trajectories and stacks them into NumPy arrays.

        Returns:
            tuple:
                - initial_h (np.ndarray): Initial recurrent states with shape
                (num_trajectories, 4096).
                - z_all (np.ndarray): Latent state sequences (logits) with shape
                (trajectory_length + 1, num_trajectories, 1024).
                - a_all (np.ndarray): Action sequences with shape
                (trajectory_length + 1, num_trajectories, 1).
        """
        if len(self) == 0: return (None, None, None)

        initial_h = np.zeros((1, len(self.trajectories), 4096), dtype=np.float32)
        z_all = np.zeros((self.trajectory_length + 1, len(self.trajectories), 1024), dtype=np.float32)
        a_all = np.zeros((self.trajectory_length + 1, len(self.trajectories), 6), dtype=np.float32)

        for i, (key, val) in enumerate(self.trajectories.items()):  # val: (TrajectoryMemory, idx, uncertainty, time of birth)

            # value
            traj_obj: TrajectoryObject = val[0]
            traj_nr: int = val[1]
            trajectory: np.array = traj_obj.get_trajectory(traj_nr)
            z_s = np.array(trajectory[:,:-6])   # ! are logits # shape (length, 1024)
            a_s = np.array(trajectory[:,-6:])    # shape(length, 

            print("get samples: a_s.shape", a_s.shape)
            print("get samples: a_all.shape", a_all.shape)

            z_all[0, i] = np.frombuffer(key[1], dtype=np.float32)  # from shape (512,) into shape (1024,)
            a_all[0, i] = np.frombuffer(key[2], dtype=np.float32) 
            z_all[1:(z_s.shape[0]+1), i] = z_s
            a_all[1:(z_s.shape[0]+1), i] = a_s
            initial_h[0, i] = np.frombuffer(key[0], dtype=np.float32)#.reshape(4096)
        # [h1, h2, ...] [zs, ...] [as, ...]
        # batch example: [h1, h2, h3]  [zs1, zs2, zs3] [as1, as2, as3]   ####### shape(sequenz, batch, ... 32, 32)
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

    def _prune_memory(self, prune_fraction: float=0.05):
        to_prune = int(len(self.trajectories) * prune_fraction)
        # recalc all uncertainties or do this in extra training?
        pass
    
    def _add_hnsw(self, key):
        if self.hnsw_storage.get_current_count() == self.hnsw_storage.get_max_elements():
            print("Problem")
        key = np.array([self._flatten_key(key).reshape(1, -1)])
        self.hnsw_storage.add_items(key)
        
    def kNN(self, key: tuple, k: int = 1):
        """Return the k-nearest neighbors among stored trajectory keys."""
        
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
        neighbors = [self.trajectories[keys[i]] for i in indices[0]]

        return neighbors
        

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
    def __init__(self, trajectory_length: int):
        """
        Docstring for __init__
        
        :param trajectory_length: Length of expected trajectory
        :type trajectory_length: int
        """
        self.trajectory_length: int = trajectory_length
        """The maximum length each trajectory will have."""
        self.free_space: int = trajectory_length
        """How much free space this object still has. Always resets if new trajectory starts."""
        self.memory: np.array = np.empty((trajectory_length, 1030), dtype=np.float32)  # TODO: add size of tuple (z_t', a_t') ~ 1024+1
        """"The actual trajectories."""

        self.traj_num_to_offset : np.array = np.zeros((10,), dtype=int) # 10 is test value for now
        """"The actual trajectory starting index."""
        self.num_trajectories : int = 0
        """"Trajectory counter."""

    def new_traj(self):
        """"Extend the internal memory, so it can hold another trajectory.

        :return: The internal number for the new trajectory in this object.
        """
        nr_idx = self.num_trajectories
        self.num_trajectories += 1

        if self.traj_num_to_offset.shape[0] <= self.num_trajectories + 1:
            self.traj_num_to_offset = np.concatenate(
                (self.traj_num_to_offset, np.zeros((10,), dtype=int)),
                axis=0
            )

        self.memory = np.concatenate((self.memory, np.empty((self.trajectory_length-self.free_space, 1030), dtype=np.float32)), axis=0) # possible if lenght-freespace = 0 ??? # TODO: add size of tuple (z_t', a_t')
        self.free_space = self.trajectory_length
        self.traj_num_to_offset[nr_idx] = self.last_idx()

        return nr_idx

    def del_traj(self, traj_nr):
        """ This function deletes a trajectory (a contiguous block of entries) from the internal memory based on a given trajectory number.
        It: Computes the start and end indices of the trajectory to remove using traj_num_to_offset.
        Removes that slice from self.memory.
        Shifts all subsequent data left to fill the gap.
        Updates traj_num_to_offset so that indices of following trajectories are decremented by the length of the deleted trajectory.
        So 'traj_nr' will now point to the start of the 'traj_nr' + 1 trajectory
        Handles edge cases such as deleting the last trajectory or an empty trajectory.
        Delete a trajectory by its number 

        :param value: internal trajectory number to delete.
        """
        start_idx = self.traj_num_to_offset[traj_nr-1]+self.trajectory_length if traj_nr-1 >=0 else 0
        end_idx = self.traj_num_to_offset[traj_nr+1] if traj_nr < self.traj_num_to_offset.shape[0]-1 else self.memory.shape[0]

        # print("DELETE TRAJ NR:", traj_nr, " FROM ", start_idx, " TO ", end_idx)

        to_delete = (end_idx - start_idx)
        if to_delete > 0:
            # if self.memory.shape[0] != end_idx:
            ## np.concatenate([array([], dtype=int64), array([2, 3, 4])], axis=0) -> array([2, 3, 4]) ~Good
            self.memory = np.concatenate([self.memory[:start_idx], self.memory[end_idx:]], axis = 0) # +1-1*1/1????
            # elif start_idx == 0:
            #     self.memory = self.memory[end_idx:]
            # else:
            #     self.memory = self.memory[:start_idx]

            # decrement following trajectory indices by length of deleted trajectory
            if traj_nr + 1 < self.num_trajectories:
                temp = np.zeros_like(self.traj_num_to_offset)
                temp[traj_nr+1:] = (end_idx - start_idx)

                self.traj_num_to_offset -= temp
            # traj_nr marks the last trajectory
            else:
                self.traj_num_to_offset[traj_nr] = start_idx
                self.free_space = 0

        else:
            self.traj_num_to_offset[traj_nr] = self.traj_num_to_offset[traj_nr+1] if traj_nr + 1 < self.num_trajectories else start_idx

    def add(self, value: np.array) -> int:
        """
        Add a value into the trajectory.
                
        :param value: The valuravel()
        :return: The remaining free space.
        """
        self.memory[-self.free_space] = value # TODO: add tuple (z_t', a_t')
        self.free_space -= 1
        # print("ADD VALUE:", value)
        # print("memory: ", type(self.memory[0]))
        # print(type(self.memory))

        # ADD VALUE: (array([[[0., 0., 0., ..., 0., 0., 0.]]], dtype=float32), array([[2]]))
        # memory:  <class 'numpy.ndarray'>
        # <class 'numpy.ndarray'>
        return self.free_space

    def last_idx(self):
        return self.memory.shape[0] - self.free_space

    def __str__(self):
        return f"TrajectoryObj| Free space: {self.free_space}| Trajectory length: {self.trajectory_length} \
            | Traj num. to offset: {self.traj_num_to_offset}"
    
    def get_trajectory(self, traj_nr) -> np.array:
        """Returns a specific trajectory in full"""
        start_idx = self.traj_num_to_offset[traj_nr-1]+self.trajectory_length if traj_nr-1 >=0 else 0
        end_idx = start_idx + self.trajectory_length
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