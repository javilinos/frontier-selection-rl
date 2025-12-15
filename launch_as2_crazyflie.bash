#!/bin/bash

usage() {
    echo "  options:"
    echo "      -r: record rosbag"
}

# Arg parser
while getopts "r" opt; do
  case "$opt" in
    r)
      record_rosbag="true"
      ;;
    :)  # missing argument for -s
      echo "Error: -$OPTARG requires a value" >&2
      usage
      ;;
    \?) # invalid option
      echo "Error: invalid option -$OPTARG" >&2
      usage
      ;;
  esac
done

# Shift optional args
shift $((OPTIND -1))

## DEFAULTS
display_rviz=true
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}

drones=($(python3 utils/get_drones.py -p cf_config/config.yml))
tmuxinator start -n aerostack2 -p tmuxinator/aerostack2_crazyflie.yml drones=${drones} &
wait

if [[ ${record_rosbag} == "true" ]]; then
  tmuxinator start -n rosbag -p tmuxinator/rosbag.yml drone_namespace=${drones} &
  wait
fi

if [[ ${launch_keyboard_teleop} == "true" ]]; then
  tmuxinator start -n keyboard_teleop -p tmuxinator/keyboard_teleop.yml \
    simulation=true \
    drone_namespace=$(python utils/get_drones.py ${simulation_config} --sep ",") &
  wait
fi

# Attach to tmux session ${drones[@]}, window mission
tmux attach-session -t aerostack2:mission
