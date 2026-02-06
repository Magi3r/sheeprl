#!/bin/bash

game="dem_100k_hero"
GPU=2

CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto seed=420 \
                                torch_use_deterministic_algorithms=true \
                                torch_backends_cudnn_benchmark=false \
                                torch_backends_cudnn_deterministic=true \
                                cublas_workspace_config=:4096:8 \
                                episodic_memory.use_episodic_memory=true \
                                episodic_memory.enable_rehearsal_training=false \
                                episodic_memory.use_acd=false \
                                episodic_memory.fill_parallel_to_buffer=false \
                                episodic_memory.adc_weighting=false \
                                # episodic_memory.capacity=1024 \
                                # episodic_memory.k_neighbors=10 \
                                # episodic_memory.rehearsal_train_every=512

## CUBLAS_WORKSPACE_CONFIG=:4096 -> some functions not deterministic, so we need to set this env_var. so it works :)
# CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dreamer_v3 fabric.accelerator=gpu fabric.strategy=auto seed=420
                                # episodic_memory.use_episodic_memory=false