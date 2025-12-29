import os
import sys

# PROJECT_ROOT = Path(__file__).resolve().parents[1]   # …/my_project
# sys.path.append(str(PROJECT_ROOT))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import torch

from episodic_memory import EpisodicMemory, TrajectoryObject
# adjust import path accordingly


# ---------------------------
# Fixtures
# ---------------------------

@pytest.fixture
def dummy_shapes():
    return {
        "z": (4,),
        "h": (8,),
        "a": (3,)
    }


@pytest.fixture
def dummy_state(dummy_shapes):
    return {
        "stoch": torch.zeros(dummy_shapes["z"]),
        "deter": torch.zeros(dummy_shapes["h"]),
    }


@pytest.fixture
def dummy_action(dummy_shapes):
    return torch.zeros(dummy_shapes["a"])


@pytest.fixture
def memory(dummy_shapes):
    return EpisodicMemory(
        trajectory_length=5,
        uncertainty_threshold=0.5,
        z_shape=dummy_shapes["z"],
        action_shape=dummy_shapes["a"],
        k_nn=3,
    )


# ---------------------------
# TrajectoryObject tests
# ---------------------------

def test_trajectory_initial_state():
    traj = TrajectoryObject(trajectory_length=4)

    assert traj.trajectory_length == 4
    assert traj.free_space == 4
    assert traj.num_trajectories == 0


def test_trajectory_add_reduces_free_space():
    traj = TrajectoryObject(trajectory_length=3)

    traj.add(("z1", "a1"))
    assert traj.free_space == 2

    traj.add(("z2", "a2"))
    assert traj.free_space == 1


def test_new_traj_increments_counter_and_resets_space():
    traj = TrajectoryObject(trajectory_length=3)

    idx = traj.new_traj()

    assert idx == 0
    assert traj.num_trajectories == 1
    assert traj.free_space == 3


def test_last_idx_monotonic():
    traj = TrajectoryObject(trajectory_length=3)

    start = traj.last_idx()
    traj.add(("z", "a"))
    after = traj.last_idx()

    assert after > start


# ---------------------------
# EpisodicMemory basic tests
# ---------------------------

def test_empty_memory_len(memory):
    assert len(memory) == 0


def test_flatten_key_returns_1d_numpy(memory, dummy_state, dummy_action):
    key = (dummy_state["deter"], dummy_state["stoch"], dummy_action)

    flat = memory.flatten_key(key)

    assert isinstance(flat, np.ndarray)
    assert flat.ndim == 1


def test_create_traj_adds_entry(memory, dummy_state, dummy_action):
    key = (dummy_state["deter"], dummy_state["stoch"], dummy_action)

    memory.create_traj(key, uncertainty=1.0)

    assert len(memory.trajectories) == 1


def test_fill_traj_requires_current_trajectory(memory):
    with pytest.raises(AssertionError):
        memory.fill_traj(("z", "a"))


# ---------------------------
# step() behavior
# ---------------------------

def test_step_creates_new_trajectory_on_high_uncertainty(
    memory, dummy_state, dummy_action
):
    memory.prev_state = dummy_state

    memory.step(
        state=dummy_state,
        action=dummy_action,
        uncertainty=1.0,  # above threshold
        done=False,
    )

    assert len(memory.trajectories) == 1
    assert memory.current_trajectory is not None


def test_step_does_not_create_traj_on_low_uncertainty(
    memory, dummy_state, dummy_action
):
    memory.prev_state = dummy_state

    memory.step(
        state=dummy_state,
        action=dummy_action,
        uncertainty=0.1,
        done=False,
    )

    assert len(memory.trajectories) == 0
    assert memory.current_trajectory is None


def test_step_done_resets_current_trajectory(
    memory, dummy_state, dummy_action
):
    memory.prev_state = dummy_state

    memory.step(
        state=dummy_state,
        action=dummy_action,
        uncertainty=1.0,
        done=True,
    )

    assert memory.current_trajectory is None
    assert memory.prev_state is None


# ---------------------------
# kNN interface tests
# ---------------------------

def test_knn_empty_returns_empty(memory, dummy_state, dummy_action):
    key = (dummy_state["deter"], dummy_state["stoch"], dummy_action)

    neighbors = memory.kNN(key, k=3)

    assert neighbors == []


def test_knn_returns_at_most_k(memory, dummy_state, dummy_action):
    for _ in range(5):
        key = (
            torch.randn_like(dummy_state["deter"]),
            torch.randn_like(dummy_state["stoch"]),
            torch.randn_like(dummy_action),
        )
        memory.create_traj(key, uncertainty=1.0)

    query_key = (
        dummy_state["deter"],
        dummy_state["stoch"],
        dummy_action,
    )

    neighbors = memory.kNN(query_key, k=2)

    assert len(neighbors) <= 2


def test_knn_returns_trajectory_tuples(memory, dummy_state, dummy_action):
    key = (dummy_state["deter"], dummy_state["stoch"], dummy_action)
    memory.create_traj(key, uncertainty=1.0)

    neighbors = memory.kNN(key, k=1)

    traj, offset, uncertainty = neighbors[0]
    assert isinstance(traj, TrajectoryObject)
    assert isinstance(offset, int)
    assert isinstance(uncertainty, float)


# ---------------------------
# String representations
# ---------------------------

def test_memory_str_contains_summary(memory):
    s = str(memory)

    assert "Num trajectories" in s
    assert "Trajectory length" in s


def test_trajectory_str_does_not_crash():
    traj = TrajectoryObject(trajectory_length=3)
    s = str(traj)

    assert isinstance(s, str)