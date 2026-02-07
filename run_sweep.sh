#!/bin/bash

game="dem_100k_hero"
num_parallel=2
GPUS=( 0 3)

i=0

# for acd in true false; do
for em_size in 2048 256; do
  sleep 5
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}
  # echo "CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.adc_weighting=$acd"
  # sleep $((RANDOM % 20))
  echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.capacity=$em_size episodic_memory.k_neighbors=20"
  i=$((i+1))
  sleep 5

done | parallel -j $num_parallel