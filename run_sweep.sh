#!/bin/bash

game="dem_100k_seaquest"
num_parallel=4
GPUS=(0 1 2 3)

i=0
for k in 10 20; do
  for size in 256 2048; do
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    echo "CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.capacity=$size episodic_memory.k_neighbors=$k"
    i=$((i+1))
    sleep 5
  done
  sleep 5
done | parallel -j $num_parallel