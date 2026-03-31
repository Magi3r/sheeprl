apt update
apt install -y libgl1 libglib2.0-bin # Required.
apt install -y parallel # Not needed, used in run_sweep.sh
apt install -y tmux # Not needed, used for background execution while we are disconnected from the servers