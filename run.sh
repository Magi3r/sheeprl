exp=$1
cuda_device=$2

if [ -z "$exp" ]; then
  echo "Usage: ./run.sh <exp_name> [cuda_device]"
  echo "Example: ./run.sh dem_100k_ms_pacman 0"
  exit 1
fi

if [ -z "$cuda_device" ]; then
  cuda_device=0
fi

# tmux new -s dreamer_$game "echo "======  DETACH WITH Ctrl+b d  ======" && CUDA_VISIBLE_DEVICES=$cuda_device uv run dreamer.py --configs atari100k --task atari_$game --logdir ./logdir/atari_$game"
# CUDA_VISIBLE_DEVICES=$cuda_device uv run sheeprl.py exp=$exp algo=dem fabric.accelerator=gpu episodic_memory.capacity=512 episodic_memory.fill_parallel_to_buffer=true episodic_memory.normalize_kNN=false -m hydra/launcher=submitit_local #fabric.strategy=ddp fabric.devices=1
uv run sheeprl.py exp=$exp algo=dem fabric.accelerator=gpu fabric.strategy=auto episodic_memory.enable_rehersal_training=false episodic_memory.use_acd=false episodic_memory.fill_parallel_to_buffer=false episodic_memory.normalize_kNN=false episodic_memory.softmax_kNN=true  