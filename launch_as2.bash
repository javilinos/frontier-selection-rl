#!/bin/bash

usage() {
    echo "  options:"
    echo "      -m: multi agent"
    echo "      -r: record rosbag"
    echo "      -t: launch keyboard teleoperation"
}

# Arg parser
while getopts "mrt" opt; do
  case ${opt} in
    m )
      swarm="true"
      ;;
    r )
      record_rosbag="true"
      ;;
    t )
      launch_keyboard_teleop="true"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

# Shift optional args
shift $((OPTIND -1))

export GZ_SIM_RESOURCE_PATH=$PWD/assets/worlds:$PWD/assets/models:$GZ_SIM_RESOURCE_PATH

## DEFAULTS
display_rviz=true
swarm=${swarm:="false"}
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}

simulation_config="assets/worlds/world_low_density.json" 
if [[ ${swarm} == "true" ]]; then
  simulation_config="assets/worlds/world3.json"
fi

drones=($(python3 utils/get_drones.py ${simulation_config} --sep ' '))
echo drones: ${drones[@]}
for drone in "${drones[@]}"; do
  echo "Starting tmuxinator session for drone: ${drone}"
  tmuxinator start -n ${drone} -p tmuxinator/session.yml \
    drone_namespace=${drone} \
    simulation_config=${simulation_config} &
  wait
done

# if [[ ${record_rosbag} == "true" ]]; then
#   tmuxinator start -n rosbag -p tmuxinator/rosbag.yml \
#     drone_namespace=$(python utils/get_drones.py ${simulation_config}) &
#   wait
# fi

# if [[ ${launch_keyboard_teleop} == "true" ]]; then
#   tmuxinator start -n keyboard_teleop -p tmuxinator/keyboard_teleop.yml \
#     simulation=true \
#     drone_namespace=$(python utils/get_drones.py ${simulation_config} --sep ",") &
#   wait
# fi

tmuxinator start -n swarm_nodes -p tmuxinator/session_swarm.yml \
  simulation_config=${simulation_config} &
wait

tmuxinator start -n gazebo -p tmuxinator/gazebo.yml \
  simulation_config=${simulation_config} \
  display_rviz=${display_rviz} &
wait

# Attach to tmux session ${drones[@]}, window mission
tmux attach-session -t ${drones[0]}:mission
