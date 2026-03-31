import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # disable eager execution
import time
import shutil

# Define the path to the logs directory
# logs_dir = "./logs/runs/dem/BreakoutNoFrameskip-v4"
# logs_dir = "./logs/runs/dem/HeroNoFrameskip-v4/"
logs_dir = "./logs/runs/dem/SeaquestNoFrameskip-v4/"

print("\n\nscanning:", logs_dir)

deleting_files = []

# Iterate over all subdirectories in the logs directory
for run_dir in os.listdir(logs_dir):
    run_dir_path = os.path.join(logs_dir, run_dir)

    # Check if the subdirectory is a run directory (i.e., it contains a 'version_0' directory)
    if os.path.isdir(run_dir_path) and 'version_0' in os.listdir(run_dir_path):
        version_0_dir = os.path.join(run_dir_path, 'version_0')

        # # also remove run_dir_path folder if version_0_dir is empty
        if not os.listdir(version_0_dir):
            deleting_files.append(run_dir_path)
            continue

        # Iterate over all files in the 'version_0' directory
        for file in os.listdir(version_0_dir):
            file_path = os.path.join(version_0_dir, file)
            
            # Check if the file is an events file (i.e., it ends with '.tfevents.')
            if '.tfevents.' in file:
                # Try to read the events file
                try:
                    for event in tf.compat.v1.train.summary_iterator(file_path):
                        # Check if the event is a scalar summary (i.e., it has a 'value' field)
                        if event.summary:
                            # Check if the scalar summary has a 'step' field and its value is greater than 1000
                            if event.step > 1000:
                                # If the step is greater than 1000, break out of the loop
                                print("skipping: ", run_dir_path)
                                break
                    else:
                        # If the loop completes without finding a step greater than 1000, delete the run directory
                        deleting_files.append(run_dir_path)
                        # print(f"will deleted run directory: {run_dir_path}")
                except Exception as e:
                    # If there's an error reading the events file, print the error and continue to the next file
                    print(f"\nError reading events file: {file_path}. Error: {e}")

# Calculate total size of all directories to be deleted
total_size_bytes = 0
for dir_path in deleting_files:
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size_bytes += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    continue
def format_bytes(bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"
print(f"\nTotal storage to be freed: {format_bytes(total_size_bytes)}")

deleting_files.sort()

my_input = ""
print(f"\ncan remove: {len(deleting_files)} files with steps <= 1000")
if deleting_files:
    if len(deleting_files) >= 10:
        for i, file in enumerate(deleting_files):
            print(f"    {i}th: {deleting_files[i]}")
            if i > 9: break
    print(f"    last: {deleting_files[-1]}")

    my_input = input(f"U sure u want to continuous deleting all {len(deleting_files)} files (y/n)?")

if my_input.lower() in ["yes", "y", "ja", "j"]:
    print("deleting...")
    for dir in deleting_files:
        shutil.rmtree(dir, ignore_errors=True)
        pass
    print("done!")
else:
    print("nothing happens...")