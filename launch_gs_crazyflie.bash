#!/bin/bash

usage() {
    echo "  options:"
    echo "      -m: disable launch mocap4ros2. By default set."
    echo "      -t: launch keyboard teleoperation. By default not set."
    echo "      -v: launch rviz. By default not set."
    echo "      -r: record rosbag. By default not set."
}

# Initialize variables with default values
mocap4ros2="false"
keyboard_teleop="false"
rviz="false"
rosbag="false"
use_gnome="false"

# Parse command line arguments
while getopts "mtvr" opt; do
  case ${opt} in
    m )
      mocap4ros2="true"
      ;;
    t )
      keyboard_teleop="true"
      ;;
    v )
      rviz="true"
      ;;
    r )
      rosbag="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[swrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

# If no drone namespaces are provided, get them from the world description config file
drones_namespace_comma=($(python3 utils/get_drones.py -p cf_config/config.yml --sep ','))
IFS=',' read -r -a drone_namespaces <<< "$drones_namespace_comma"

# Launch aerostack2 ground station
tmuxinator start -n ground_station -p tmuxinator/ground_station_crazyflie.yml \
  drone_namespace=${drones_namespace_comma} \
  keyboard_teleop=${keyboard_teleop} \
  rviz=${rviz} \
  rosbag=${rosbag} \
  mocap4ros2=${mocap4ros2} &
wait

# Attach to tmux session
tmux attach-session -t ground_station