#!/bin/bash

games=(
    # "dem_100k_ms_pacman" 
    # "dem_100k_hero" 
    # "dem_100k_pong"
    # "dem_100k_space_invaders"
    "dem_100k_demon_attack"
    "dem_100k_breakout"
)

for item in "${games[@]}"; do
    ./run.sh $item "3"
done
