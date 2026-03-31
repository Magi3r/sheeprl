import os
import glob
import yaml
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.contrib.concurrent import thread_map


"""
General Idea:
    - find every tensorboard (tfevents) file inside the logs* folder(s) | we use glob for this
    - each tfevents file has a corresponding overrides.yaml in which the command line arguments are stated
        - (we controlled our parmeters though this)
    - Extract the reward from the tfevents files and group the reward-arrays with a same config (overrides) togther inside a dictionary
        - now we got dictionary (config_groups) with a config (overrides.yaml) as key and a list of rewardsarrays as value.

For each plot now we do the following:
    merge the values from 'config_groups' that has the same parameters we are interested in, and discard all other configs:
    - create a filter-dictionary for params we are interested in:
        eg:
            "episodic_memory.capacity": "1024", # only take configs that have this value
            "episodic_memory.use_acd": None, # the config needs this key, we want grouping based on distinct values
        if we dont include a key in the dictionary, we dont care whether it exist, or the value it has.
    - We then "run" the filter over all configs in 'config_groups' and merge/discard the value accordingly.

    - Then average all reward arrays after the filtering that are in the same group and plot the results.
"""

# ---------------------------
# Settings
# ---------------------------
# LOG_ROOT = "logs/runs" # /dem/HeroNoFrameskip-v4"
LOG_ROOT = "logs*/runs" # /dem/HeroNoFrameskip-v4"
OVERRIDES_FILE = ".hydra/overrides.yaml"
# OVERRIDES_FILE = ".hydra/config.yaml" ## does not work, since here we have other yaml structure
METRIC_NAME = "Rewards/rew_avg"  # Replace with your metric
# ---------------------------

DEFAULT_PARAMS = {
  "algo": "dem",
  "exp": "dem_100k_hero",
  "episodic_memory.use_episodic_memory": "true",
  "episodic_memory.trajectory_length": "20",
  "episodic_memory.capacity": "1024",
  "episodic_memory.k_neighbors": "5",
  "episodic_memory.uncertainty_threshold": "0.9",   #legacy no longer used, as dynamic calculated
  "episodic_memory.prune_fraction": "0.2",
  "episodic_memory.time_to_live": "100",
  "episodic_memory.rehearsal_train_every": "512",
  "episodic_memory.enable_rehearsal_training": "true",
  "episodic_memory.use_acd": "true",
  "episodic_memory.fill_parallel_to_buffer": "false" ,
  "episodic_memory.normalize_kNN": "false",
  "episodic_memory.softmax_kNN": "false",
  "episodic_memory.adc_weighting": "true",
  "episodic_memory.replace_by_acd": "false",
  "episodic_memory.std_multiplier": "1.0"
}
# ----------------------

def load_overrides(run_path):
    """Load Hydra overrides.yaml as a hashable tuple."""
    yaml_path = os.path.join(run_path, OVERRIDES_FILE)

    if not os.path.exists(yaml_path):
        return None

    with open(yaml_path, "r") as f:
        overrides = yaml.safe_load(f)
    # Convert list of overrides to a sorted tuple for hashing
    # e.g., ['+lr=0.001', '+batch_size=32'] -> tuple sorted
    filtered = [o for o in overrides if "seed=" not in o.lower()]
    return tuple(sorted(filtered))
    # return tuple(sorted(overrides))

def extract_metric(ef, metric_name):
    """Extract scalar metric (rewards) from TensorBoard events."""
    # event_files = glob.glob(os.path.join(run_path, "version_0", "events.out.tfevents.*"))
    # print(event_files)

    ea = EventAccumulator(ef)
    ea.Reload()

    values = [0.0]
    steps = [0]
    last_step = steps[0]
    step_thresh = 90_000

    if metric_name in ea.Tags().get('scalars', []):
        step_size = 1000
        first_time = ea.Scalars(metric_name)[0].wall_time
        last_time = ea.Scalars(metric_name)[-1].wall_time
        hours = (last_time - first_time) / 3600
        if hours < 4: # filter out runs, with "original" dreamer parameter (runs were like ~2.5h), otherwise shortest runs 7h+
            return None
        for e in ea.Scalars(metric_name):
            current_step = e.step
            if current_step > last_step + step_size:
                # do repeated
                fraction = step_size / (current_step - last_step)
                for i, step in enumerate(range(last_step + step_size, current_step, step_size)): # interpolate values at steps between two rewards
                    values.append(values[-1] + (e.value - values[-1]) * fraction * (i+1))
                    steps.append(step)
            values.append(e.value)
            steps.append(current_step)
            last_step = current_step

        if current_step < step_thresh: # run not long enough to include in the results
            return None

        if current_step != 100_000: # some runs dont have a reward at exactly 100_000 (last episode not finished), so just fill
            for step in range(current_step, 100_000, step_size):
                values.append(values[-1])
                steps.append(step + step_size)
    
        return values

    return None


def dict_to_tuple(d):
    return tuple(sorted(f"{k}={v}" for k, v in d.items()))

def tuple_to_dict(tup):
    """
    Converts a tuple of strings like 'key=value' into a dictionary.
    """
    result = {}
    for item in tup:
        if '=' in item:
            key, value = item.split('=', 1)  # split only at the first '='
            result[key] = value.lower()
    return result

def change_grouping(config_groups: dict[tuple, list], filter: dict):
    """
        filter (dict): which keys need to exist in config and match val 
        (None = param dont neet to be included, "" = value ignored)
    """
    
    def filter_dict(d1, d2):
        return {k: v for k, v in d1.items() if k in d2}

    filtered = {}

    def threading_function(cfg):
        #do the filtering
        filtered_conf = {}
        insert = True
        cfg_dict = tuple_to_dict(cfg)
        if "episodic_memory.capacity" in cfg_dict and cfg_dict["episodic_memory.capacity"] == "4096": #too few runs with 4096 to include
            insert = False
        cfg_dict = filter_dict(cfg_dict, filter) #remove keys from cfg_dict that are not in filter
        for kii, val in filter.items():
            # insert default values
            if kii not in cfg_dict: # insert default values if key not in cfg_dict
                if cfg_dict["algo"] == "dreamer_v3":
                    continue
                cfg_dict[kii] = DEFAULT_PARAMS[kii]
            if val is not None and val.lower() != cfg_dict[kii].lower(): # check if wrong value for filter
                insert = False
                break
                
        if insert:
            new_cfg = dict_to_tuple(cfg_dict)
            filtered_conf[new_cfg] = []
            filtered_conf[new_cfg].extend(config_groups[cfg])
        
        return filtered_conf
    
    for cfg in thread_map(threading_function, config_groups):
        for key in cfg.keys():
            if key not in filtered:
                filtered[key] = []
            filtered[key].extend(cfg[key])

    return filtered


def make_plot(
    config_groups,
    our_filter,
    metric_name,
    save_path,
    title=None,
    order_fn=None,
    label_fn=None,
    figsize=(8, 5),
    linestyles=None,
    add_std = True
):
    # ---------------------------
    # Filter + regroup
    # ---------------------------
    config_groups_filtered = change_grouping(config_groups, our_filter)

    # ---------------------------
    # Compute stats
    # ---------------------------
    results = {}
    for config, data_list in config_groups_filtered.items():
        print(f"len(data_list): {len(data_list)}")
        if len(data_list) == 0:
            continue
        all_runs = np.vstack(data_list)
        mean_metric = np.mean(all_runs, axis=0)
        std_metric = np.std(all_runs, axis=0)
        results[config] = {"mean": mean_metric, "std": std_metric}

    print(f"len results (plots): {len(results)}")

    # ---------------------------
    # Optional ordering
    # ---------------------------
    items = list(results.items())
    if order_fn is not None:
        items = sorted(items, key=lambda x: order_fn(tuple_to_dict(x[0]), x[1])) # we only ever use x[0] for sorting

    # ---------------------------
    # Plot
    # ---------------------------
    plt.figure(figsize=figsize)
    colors = plt.get_cmap("tab10") #, len(items))

    if linestyles is None or len(linestyles) != len(items):
        linestyles = ["solid"] * len(items) 

    print(f"Plot: {title} - {metric_name}")

    for i, (config, stats) in enumerate(items):
        color = colors(i)
        mean = stats["mean"]
        std = stats["std"]
        steps = np.arange(len(mean))

        cfg_dict = tuple_to_dict(config)

        # ---------------------------
        # Label handling
        # ---------------------------
        if label_fn is not None:
            label = label_fn(cfg_dict, stats)
        else:
            # default behavior (your current logic)
            config_label = {
                k: v
                for k, v in cfg_dict.items()
                if k in our_filter.keys() and our_filter[k] is None
            }

            config_label = {
                (k.split(".")[-1] if k is not None else "None"): v
                for k, v in config_label.items()
            }

            label = str(config_label)
        plt.plot(steps*1000, mean, label=label,color=color, linestyle=linestyles[i])
        if add_std:
            plt.fill_between(steps*1000, mean - std, mean + std, alpha=0.2, color=color, linestyle=linestyles[i])

        print(f"{label} - Final mean score: {mean[-1]} - Final standard deviation: {std[-1]}")

    plt.xlabel("Step")
    plt.ylabel("Avg. Reward")

    if title is None:
        title = our_filter.get("exp", "experiment")
    if isinstance(title, str) and "dem_" in title:
        title = title.split("_")[-1]

    plt.title(f"{title}")
    plt.xlim(0,100000)

    plt.legend()
    plt.tight_layout(pad=0.0)
    plt.savefig(f"figures/figure_{save_path}.png")
    plt.savefig(f"figures/figure_{save_path}.pdf")
    plt.close()

# -----------------------------
config_groups = {}

out_files = sorted(glob.glob(os.path.join(LOG_ROOT, "**", "*.tfevents.*"), recursive=True))
print(f"found {len(out_files)} files")

filtered_files = []

# define your date range
# runs before 2026.2.6 had a logical flaw in the EM
start_date = datetime(2026, 2, 6) # Y/M/D
end_date   = datetime(2026, 2, 17)

#runs in this timeframe had an error in the configs due to weights and bias (probably all had the same config but not sure)
start_date2 = datetime(2026, 3, 2) # Y/M/D
end_date2   = datetime(2026, 4, 1)

for f in out_files:
    ctime = datetime.fromtimestamp(os.path.getctime(f))
    if "dreamer_v3" in f:   # if algo is dreamer_v3 then its a baseline run
        filtered_files.append(f)
        continue
    if start_date <= ctime <= end_date or start_date2 <= ctime <= end_date2:
        filtered_files.append(f)

out_files = filtered_files
print(f"found filtered date {len(out_files)} files")

# for run_folder in os.listdir(LOG_ROOT):
for file in out_files:
    # run_path = os.path.join(LOG_ROOT, run_folder)
    run_path = os.path.dirname(os.path.dirname(file))
    config = load_overrides(run_path)
    if config is None:
        continue
    metric_data = extract_metric(file, METRIC_NAME) # run_path
    if metric_data is None:
        continue
    if config not in config_groups:
        config_groups[config] = []
    config_groups[config].append(metric_data)
    

#------------------------------------------------------------------------------------------------
# H.E.R.O. Baseline Graph
#------------------------------------------------------------------------------------------------ 

def label_baseline_hero(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"
    return label

def order_baseline_hero(e1, e2):
    if e1["algo"] == "dem":
        return 1
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": None, 
        "exp": "dem_100k_hero",
        "episodic_memory.use_acd": None,
        "episodic_memory.replace_by_acd": "false",
    },
    metric_name=METRIC_NAME,
    save_path="hero_baseline",
    title="DEM vs Baseline, H.E.R.O",
    label_fn=label_baseline_hero,
    order_fn=order_baseline_hero
)

#------------------------------------------------------------------------------------------------
# H.E.R.O. Capacity
#------------------------------------------------------------------------------------------------ 

def label_capacity_hero(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"

        label += f", Capacity={cfg_dict['episodic_memory.capacity']}"
    return label

def order_capacity_hero(e1, e2):
    if e1["algo"] == "dem":
        return int(e1["episodic_memory.capacity"])
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": None,
        "exp": "dem_100k_hero",
        "episodic_memory.capacity": None,
        "episodic_memory.use_acd": "false",
    },
    metric_name=METRIC_NAME,
    save_path="hero_capacity",
    title="DEM (no ACD) per Capacity, H.E.R.O",
    label_fn=label_capacity_hero,
    order_fn=order_capacity_hero,
    linestyles=["dashed", "solid", "solid", "solid", "solid"]
)

#------------------------------------------------------------------------------------------------
# H.E.R.O. Replace ACD vs ACD
#------------------------------------------------------------------------------------------------ 

def label_acd_compare_hero(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"

        if "episodic_memory.replace_by_acd" in cfg_dict and cfg_dict["episodic_memory.replace_by_acd"] == "true":
            label += " (Replace ACD)"
        # label += f", Capacity={cfg_dict['episodic_memory.capacity']}"
    return label

def order_acd_compare_hero(e1, e2):
    if e1["algo"] == "dem":
        return 1
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": "dem",
        "exp": "dem_100k_hero",
        "episodic_memory.use_acd": None,
        "episodic_memory.replace_by_acd": None,
        "episodic_memory.enable_rehearsal_training": "true",
    },
    metric_name=METRIC_NAME,
    save_path="hero_acd",
    title="ACD vs Replace, H.E.R.O",
    label_fn=label_acd_compare_hero,
    order_fn=order_acd_compare_hero
)

#------------------------------------------------------------------------------------------------
# Breakout Baseline
#------------------------------------------------------------------------------------------------ 

def label_baseline_breakout(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"

        if "episodic_memory.replace_by_acd" in cfg_dict and cfg_dict["episodic_memory.replace_by_acd"] == "true":
            label += " (Replace ACD)"
        # label += f", Capacity={cfg_dict['episodic_memory.capacity']}"
    return label

def order_baseline_breakout(e1, e2):
    if e1["algo"] == "dem":
        return 1
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": None,
        "exp": "dem_100k_breakout",
        "episodic_memory.use_acd": None,
        "episodic_memory.replace_by_acd": "false",
        "episodic_memory.enable_rehearsal_training": "true",
    },
    metric_name=METRIC_NAME,
    save_path="breakout_baseline",
    title="DEM vs Baseline, Breakout",
    label_fn=label_baseline_breakout,
    order_fn=order_baseline_breakout,
)

#------------------------------------------------------------------------------------------------
# H.E.R.O. All ACD Capacities
#------------------------------------------------------------------------------------------------ 

def label_capacity_acd_hero(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"
        if cfg_dict["episodic_memory.replace_by_acd"] == "true":
            label+= " (Replace)"

        label += f", Capacity={cfg_dict['episodic_memory.capacity']}"
    return label

def order_capacity_acd_hero(e1, e2):
    if e1["algo"] == "dem":
        return int(e1["episodic_memory.capacity"])
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": None,
        "exp": "dem_100k_hero",
        "episodic_memory.capacity": None,
        "episodic_memory.use_acd": "true",
        "episodic_memory.replace_by_acd": None,
        "episodic_memory.enable_rehearsal_training": "true",
    },
    metric_name=METRIC_NAME,
    save_path="hero_all_acd",
    title="All ACD,  H.E.R.O",
    label_fn=label_capacity_acd_hero,
    order_fn=order_capacity_acd_hero,
)


#------------------------------------------------------------------------------------------------
# Breakout all reusults
#------------------------------------------------------------------------------------------------ 

def label_all_breakout(cfg_dict, stats):
    label = ""
    if cfg_dict["algo"] == "dreamer_v3":
        label+="Dreamer V3"
    else:
        label+="DEM"
        if "episodic_memory.use_episodic_memory" not in cfg_dict: # if algo=dem and no use_episodic_memory in override, then default is true
            # default true if not in overrides
            cfg_dict["episodic_memory.use_acd"] == "true"
        if cfg_dict["episodic_memory.use_acd"] == "false":
            label += " (No ACD)"
        if cfg_dict["episodic_memory.replace_by_acd"] == "true":
            label+= " (Replace)"

        label += f", Capacity={cfg_dict['episodic_memory.capacity']}"
    return label

def order_all_breakout(e1, e2):
    if e1["algo"] == "dem":
        return int(e1["episodic_memory.capacity"])
    else:
        return 0

make_plot(
    config_groups,
    our_filter = {
        "algo": None,
        "exp": "dem_100k_breakout",
        "episodic_memory.capacity": None,
        "episodic_memory.use_acd": None,
        "episodic_memory.replace_by_acd": None, 
    },
    metric_name=METRIC_NAME,
    save_path="breakout_all",
    title="All, Breakout",
    label_fn=label_all_breakout,
    order_fn=order_all_breakout,
)