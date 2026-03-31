#!/bin/bash

game="dem_100k_breakout"
# game="dem_100k_hero"

# game="dem_100k_seaquest"
# game="dem_100k_demon_attack"
# game="dem_100k_ms_pacman"
# game="dem_100k_pong"
# game="dem_100k_space_invaders"
num_parallel=1
GPUS=(1 2)

i=0

for acd in true false; do
for seed in 0 0; do # 2 3; do
  sleep 5
  GPU=${GPUS[$((i % ${#GPUS[@]}))]}

  for acd in True; do # False

    for capa in 512; do # 2048

      echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=$capa episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=False episodic_memory.prune_fraction=0.3 episodic_memory.replace_by_acd=False episodic_memory.std_multiplier=0.7068871218125461 episodic_memory.use_acd=$acd seed=$seed"


      # if [[ $i == 0  || $i == 3 ]]; then
      #   echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=2048 episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=True episodic_memory.prune_fraction=0.3 episodic_memory.replace_by_acd=True episodic_memory.std_multiplier=0.7068871218125461 episodic_memory.use_acd=True seed=$seed"
      #   # echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=2048 episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=True episodic_memory.prune_fraction=0.37150216101459743 episodic_memory.replace_by_acd=False episodic_memory.std_multiplier=0.7068871218125461 episodic_memory.use_acd=True seed=$seed"
      # else 
      #   echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=512 episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=True episodic_memory.prune_fraction=0.3 episodic_memory.replace_by_acd=True episodic_memory.std_multiplier=0.7068871218125461 episodic_memory.use_acd=True seed=$seed"
      #   # echo "sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=$GPU uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=512 episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=False episodic_memory.prune_fraction=0.8693627281147646 episodic_memory.replace_by_acd=False episodic_memory.std_multiplier=0.8072195310557349 episodic_memory.use_acd=False seed=$seed"
      # fi
    done
  done

  i=$((i+1))
  sleep 5


done | parallel -j $num_parallel

seed=0
sleep $((RANDOM % 10)) && CUDA_VISIBLE_DEVICES=1 uv run sheeprl.py exp=$game algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.use_episodic_memory=true episodic_memory.capacity=256 episodic_memory.adc_weighting=True episodic_memory.fill_parallel_to_buffer=False episodic_memory.prune_fraction=0.3 episodic_memory.replace_by_acd=True episodic_memory.std_multiplier=0.7068871218125461 episodic_memory.use_acd=True seed=$seed
