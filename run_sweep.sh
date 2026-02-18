#!/bin/bash

game="dem_100k_hero"
num_parallel=1
GPUS=( 0)

i=0

# for acd in true false; do
for seed in 0 1; do
  sleep 5
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}
  # echo "CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.adc_weighting=$acd"
  # sleep $((RANDOM % 20))
    ## fabric.accelerator=gpu fabric.strategy=auto
  echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game episodic_memory.use_episodic_memory=true episodic_memory.enable_rehearsal_training=true episodic_memory.use_acd=true algo=dem seed=$seed"
  i=$((i+1))
  sleep 5

done | parallel -j $num_parallel