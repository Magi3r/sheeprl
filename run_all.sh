#!/bin/bash

games=(
    # "dem_100k_ms_pacman" 
    # "dem_100k_hero" 
    # "dem_100k_demon_attack"
    # "dem_100k_breakout"
    # "dem_100k_pong"
    # "dem_100k_space_invaders"
    "dem_100k_seaquest"
)

for item in "${games[@]}"; do
    ./run.sh $item "0,1,2,3"
done
