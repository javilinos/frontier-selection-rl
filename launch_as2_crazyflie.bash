#!/bin/bash

drones=($(python3 utils/get_drones.py -p cf_config/config.yml))
tmuxinator start -n aerostack2 -p tmuxinator/aerostack2_crazyflie.yml drones=${drones} &
wait

# Attach to tmux session ${drones[@]}, window mission
tmux attach-session -t aerostack2:mission
