#!/bin/bash

usage() {
    echo "  options:"
    echo "      -s {low_density|medium_density|high_density}"
    echo "      -r: record rosbag"
}

# Arg parser
while getopts "s:r" opt; do
  case "$opt" in
    s)
      case "$OPTARG" in
        low_density|medium_density|high_density)
          simulation_config="assets/worlds/world_${OPTARG}.json"
          ;;
        *)
          echo "Error: invalid density '$OPTARG'" >&2
          usage
          ;;
      esac
      ;;
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

export GZ_SIM_RESOURCE_PATH=$PWD/assets/worlds:$PWD/assets/models:$GZ_SIM_RESOURCE_PATH

export AS2_EXTRA_DRONE_MODELS=quadrotor_multi_ranger:$AS2_EXTRA_DRONE_MODELS

## DEFAULTS
display_rviz=true
swarm=${swarm:="false"}
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}

drones=($(python3 utils/get_drones.py -p ${simulation_config} --sep ' '))
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
