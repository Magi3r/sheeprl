# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "memory_profiler",
#   "tqdm",
#   "scikit-learn"
# ]
# ///

import time
import numpy as np
import torch
from collections import defaultdict
import sys
import os
from memory_profiler import profile
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from episodic_memory import EpisodicMemory

def fake_state(z_dim=32, h_dim=32):
    return {
        "stoch": torch.randn(z_dim),
        "deter": torch.randn(h_dim),
    }

def fake_action(a_dim=8):
    return torch.randn(a_dim)

def fake_uncertainty(p_high=0.1):
    """With probability p_high produce high uncertainty"""
    return np.random.rand() if np.random.rand() > p_high else 1.0

@profile
def benchmark_em(
    steps=100_000,
    trajectory_length=10,
    uncertainty_threshold=0.8,
    z_dim=32,
    h_dim=32,
    a_dim=8,
    done_prob=0.01,
):
    em = EpisodicMemory(
        trajectory_length=trajectory_length,
        uncertainty_threshold=uncertainty_threshold,
        z_shape=(z_dim,),
        action_shape=(a_dim,),
        k_nn=5,
    )

    stats = defaultdict(list)

    start_time = time.perf_counter()

    for t in tqdm(range(steps), desc="EM step operations"):
        state = fake_state(z_dim, h_dim)
        action = fake_action(a_dim)
        uncertainty = fake_uncertainty()
        done = np.random.rand() < done_prob

        t0 = time.perf_counter()
        em.step(state, action, uncertainty, done)
        stats["step_time"].append(time.perf_counter() - t0)

    total_time = time.perf_counter() - start_time

    stats["total_time"] = total_time
    stats["steps"] = steps
    stats["steps_per_sec"] = steps / total_time
    stats["num_trajectories"] = len(em.trajectories)

    return em, stats

@profile
def benchmark_deletion(em: EpisodicMemory):
    if len(em.trajectories) == 0:
        return {"deletions": 0}

    keys = list(em.trajectories.keys())
    delete_times = []
    np.random.shuffle(keys)
    
    for key in tqdm(keys[: len(keys)//2], desc="EM delete operations"):
        t0 = time.perf_counter()
        em.remove_traj(key)
        delete_times.append(time.perf_counter() - t0)

    return {
        "deletions": len(delete_times),
        "avg_delete_time": np.mean(delete_times),
        "max_delete_time": np.max(delete_times),
    }

def validate_em(em: EpisodicMemory):
    errors = []

    for key, (traj_obj, idx, unc) in em.trajectories.items():
        if traj_obj.free_space < 0:
            errors.append("Negative free_space")

        if idx >= traj_obj.num_trajectories:
            errors.append("Invalid trajectory index")

        # Check offsets monotonicity
        offsets = traj_obj.traj_num_to_offset[:traj_obj.num_trajectories]
        if not np.all(np.diff(offsets) >= 0):
            errors.append("Non-monotonic trajectory offsets")

    return errors

def benchmark_knn(
    em: EpisodicMemory,
    num_keys=10_000,
    z_dim=32,
    h_dim=32,
    a_dim=8,
    k=5
):
    # generate fake keys
    keys = [
        (
            torch.randn(z_dim),
            torch.randn(h_dim),
            torch.randn(a_dim)
        )
        for _ in range(num_keys)
    ]
    
    t0 = time.perf_counter()
    for key in tqdm(keys, desc="kNN operations"):
        em.kNN(key, k)
    kNN_time = time.perf_counter() - t0
    
    return {
        "num_keys": num_keys,
        "dim": (z_dim,h_dim,a_dim),
        "k": k,
        "total_time": kNN_time
    }
    

if __name__ == "__main__":
    em, stats = benchmark_em(steps=500_000, uncertainty_threshold=0.0)

    print("=== EM Benchmark ===")
    print(f"Steps: {stats['steps']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Steps/sec: {stats['steps_per_sec']:.0f}")
    print(f"Stored trajectories: {stats['num_trajectories']}")
    
    # kNN_stats = benchmark_knn(em, num_keys=100_000, k=5)
    # print("\n=== kNN Benchmark ===")
    # print(f"Num keys: {kNN_stats['num_keys']}")
    # print(f"Total time: {kNN_stats['total_time']:.2f}s")
    # print(f"Steps/sec: {kNN_stats['num_keys']/kNN_stats['kNN_time']:.0f}")
    # print(f"Dims: {kNN_stats['dims']}")
    # print(f"k: {kNN_stats['k']}")

    del_stats = benchmark_deletion(em)
    print("\n=== Deletion Benchmark ===")
    for k, v in del_stats.items():
        print(f"{k}: {v}")

    errors = validate_em(em)
    print("\n=== Validation ===")
    if errors:
        for e in errors:
            print("ERROR:", e)
    else:
        print("No invariant violations detected.")
