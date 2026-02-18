#!/bin/bash

game="dem_100k_hero"
# game="dem_100k_ms_pacman"
num_parallel=1
GPUS=(0)

i=0

# for acd in true false; do
for seed in 1 2; do
  sleep 5
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}
  # echo "CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.adc_weighting=$acd"
  # sleep $((RANDOM % 20))
  # echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.capacity=512 episodic_memory.use_episodic_memory=false episodic_memory.enable_rehearsal_training=false episodic_memory.use_acd=false episodic_memory.adc_weighting=false episodic_memory.fill_parallel_to_buffer=false episodic_memory.replace_by_acd=false episodic_memory.prune_fraction=0.2 algo.replay_ratio=0.125 algo.world_model.recurrent_model.recurrent_state_size=8192 algo.world_model.discrete_size=64 algo.world_model.encoder.cnn_channels_multiplier=64 algo.world_model.optimizer.lr=4.0e-05 algo.actor.optimizer.lr=4.0e-05 algo.critic.optimizer.lr=4.0e-05 algo.world_model.optimizer.eps=1.0e-20 algo.actor.optimizer.eps=1.0e-20 algo.critic.optimizer.eps=1.0e-20 seed=$seed"
  echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.capacity=512 episodic_memory.use_episodic_memory=true episodic_memory.enable_rehearsal_training=true episodic_memory.use_acd=false episodic_memory.adc_weighting=false episodic_memory.fill_parallel_to_buffer=false episodic_memory.replace_by_acd=false episodic_memory.prune_fraction=0.2 seed=$seed"
  # episodic_memory.capacity=256 episodic_memory.prune_fraction=0.8
  i=$((i+1))
  sleep 5

done | parallel -j $num_parallel