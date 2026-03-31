First, to set up the environment, we used uv. Other dependencies are found in `setup.sh`. We also provide a `devcontainer.json` that should install these dependencies automatically.
```sh
uv sync --extra atari
```

To train the model, we used the standard sheeprl way, as described by their documentation. For convenience, we created multiple scripts that allow easy tuning of hyperparameters (and which GPU to use), namely `run.sh`, a wrapper around it for multiple runs on the same gpu `run_all.sh`, and a hacky `run_sweep.sh` that parallelizes runs across multiple gpus using _GNU parallel_.
Look into `run.sh` to see the actual command used.

Our configuration and our code are in the places sheeprl expects.
The config can be found scattered across multiple folders: 
- sheeprl/configs/algo/dem.yaml
    - sets parameters for monte carlo dropout
    - inherits all other parameters from base dreamer_v3.yaml
- sheeprl/configs/exp/dem_100k_<game>.yaml
    - same for all games, with exception of which environment to use of cause
    - based on the provided dreamer_v3_100k_ms_pacman.yaml
- sheeprl/configs/exp/dem.yaml
    - most relevant file, includes all parameters for episodic memory
    - automatically importet by the game specific config

Our Code lies for the most part in `sheeprl/algos/dem`. Here is `episodic_memory_gpu.py` a reimplementation of `episodic_memory.py` and the version we used for our results. Our implementation of monte carlo dropout resides in `sheeprl/models/models.py`. In that file, we also modified the `MLP`-Class to optionally enable our monte carlo dropout.

For our plots we used the `make_plots.py`. It depends on `matplotlib`, `tensorboard` and `tqdm`. It expects a folder `figures` to exist in which it puts all the plots.

For model evaluation (and creation of videos), sheeprl provides a script that can be called using `uv run sheeprl_eval.py <path/to/model/ckpt>`. We do not provide checkpoints (2GB+ per checkpoint). Sheeprl also logs memmaps of all environment interactions (actions, rgb, ...), but we did not keep them because they used a majority of disk space (>300GB).