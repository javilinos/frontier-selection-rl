#!/usr/bin/env bash
set -euo pipefail

# Default input element
if [ $# -eq 0 ]; then
  set -- "drone0"
fi

# Base sessions to target
tmux_session_list=("keyboard_teleop" "rosbag" "mocap" "gazebo" "drone0" "drone1" "drone2" "drone3" "swarm_nodes")

# Add provided namespaces (avoid duplicates later)
for ns in "$@"; do
  tmux_session_list+=("$ns")
done

# If inside tmux session, get the current session name
current_session=""
if [[ -n "${TMUX:-}" ]]; then
  current_session="$(tmux display-message -p '#S' 2>/dev/null || true)"
fi

# Deduplicate session list (preserve order)
dedup_sessions=()
declare -A seen=()
for s in "${tmux_session_list[@]}"; do
  if [[ -z "${seen[$s]:-}" ]]; then
    dedup_sessions+=("$s")
    seen[$s]=1
  fi
done
tmux_session_list=("${dedup_sessions[@]}")

grace_seconds=2.0
poll_interval=0.2

session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}

send_ctrl_c_all_windows() {
  local session="$1"
  local windows
  mapfile -t windows < <(tmux list-windows -t "$session" -F "#{window_index}" 2>/dev/null || true)

  for w in "${windows[@]}"; do
    tmux send-keys -t "${session}:${w}" C-c 2>/dev/null || true
    sleep 0.05
  done
}

# Optional: send a "harder" interactive stop (rarely needed)
send_ctrl_backslash_all_windows() {
  local session="$1"
  local windows
  mapfile -t windows < <(tmux list-windows -t "$session" -F "#{window_index}" 2>/dev/null || true)

  for w in "${windows[@]}"; do
    tmux send-keys -t "${session}:${w}" C-\\ 2>/dev/null || true
    sleep 0.05
  done
}

wait_for_session_to_die() {
  local session="$1"
  local waited=0

  while session_exists "$session"; do
    sleep "$poll_interval"
    waited=$(python3 - <<PY
w = float("$waited") + float("$poll_interval")
print(w)
PY
)
    # If waited long enough, break
    python3 - <<PY
import sys
if float("$waited") >= float("$grace_seconds"):
  sys.exit(0)
sys.exit(1)
PY
    if [ $? -eq 0 ]; then
      break
    fi
  done
}

echo "[stop] Current tmux session: ${current_session:-<none>}"
echo "[stop] Target sessions: ${tmux_session_list[*]}"

# 1) Graceful stop: send Ctrl+C to each target session windows
for session in "${tmux_session_list[@]}"; do
  if session_exists "$session"; then
    echo "[stop] Sending Ctrl+C to session: $session"
    send_ctrl_c_all_windows "$session"
  fi
done

# 2) Give them a moment to exit on Ctrl+C
for session in "${tmux_session_list[@]}"; do
  if session_exists "$session"; then
    wait_for_session_to_die "$session"
  fi
done

# 3) Escalate: if still alive, send Ctrl+\ (SIGQUIT-ish for many CLI apps)
for session in "${tmux_session_list[@]}"; do
  if session_exists "$session"; then
    echo "[stop] Escalating (Ctrl+\\) for session: $session"
    send_ctrl_backslash_all_windows "$session"
  fi
done

# 4) Hard stop: kill sessions (but avoid killing current until the end)
for session in "${tmux_session_list[@]}"; do
  if session_exists "$session"; then
    if [[ -n "$current_session" && "$session" == "$current_session" ]]; then
      continue
    fi
    echo "[stop] Killing tmux session: $session"
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
done

# 5) Optional cleanup of stray processes (commented by default)
# echo "[stop] Killing leftover ROS/Gazebo processes..."
# pkill -TERM -f "ros2" || true
# pkill -TERM -f "rviz2" || true
# pkill -TERM -f "ros_gz_bridge" || true
# pkill -TERM -f "gz sim|gazebo|gzserver|gzclient" || true
# sleep 0.5
# pkill -KILL -f "ros2" || true
# pkill -KILL -f "rviz2" || true
# pkill -KILL -f "ros_gz_bridge" || true
# pkill -KILL -f "gz sim|gazebo|gzserver|gzclient" || true

# 6) Finally, kill current session (if inside tmux) AFTER everything else
if [[ -n "$current_session" && -n "${TMUX:-}" ]]; then
  echo "[stop] Killing current tmux session: $current_session"
  tmux kill-session -t "$current_session" 2>/dev/null || true
fi

echo "[stop] Done."
