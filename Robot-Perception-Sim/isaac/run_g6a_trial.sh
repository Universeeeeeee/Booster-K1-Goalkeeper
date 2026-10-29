#!/usr/bin/env bash
# G6-A full-scene replay; G1-G5 are read-only dependencies.
set -uo pipefail
poc=/home/lizhengran/isaacsim-workspace/g6a_head_replay
[[ $# -ge 5 ]] || { echo 'Usage: sender receiver direct_goal|one_bounce 0|10 neutral|yaw_pitch [port]' >&2; exit 2; }
sender=$1 receiver=$2 trajectory=$3 exposure=$4 head_pose=$5 port=${6:-30060}
if ! [[ "$port" =~ ^[0-9]{4,5}$ ]] || ((10#$port < 1024 || 10#$port > 65534 || 10#$port % 2)); then
  echo 'SDK stream port must be an even integer from 1024 to 65534.' >&2
  exit 2
fi
case "$trajectory" in
  direct_goal) frozen=/g4/physical_v2/direct_middle_center_repeat_0.json ;;
  one_bounce) frozen=/g4/physical_v2/bounce_one_repeat_0.json ;;
  *) exit 2 ;;
esac
[[ "$exposure" == 0 || "$exposure" == 10 ]] || exit 2
[[ "$head_pose" == neutral || "$head_pose" == yaw_pitch ]] || exit 2
for name in "$sender" "$receiver"; do
  [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
  for suffix in '' .log .exitcode; do [[ ! -e "$poc/runs/$name$suffix" ]] || exit 2; done
done
[[ "$sender" != "$receiver" ]] || exit 2
env CUDA_VISIBLE_DEVICES=GPU-50273ff3-fae7-f61c-4197-088b4d8a0037 \
  PYTHONPATH=/poc:/opt/python_deps PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  bash "$poc/run_g6a_in_ubuntu22.sh" /isaac/python.sh /poc/zed_sim_poc.py \
  --extension-root /opt/zed-v3.2.0-combined/exts --camera-usd /opt/zed-v3.2.0-combined/usd/ZED_X.usdc \
  --output "/poc/runs/$sender" --gpu 0 --render-gpu-index 0 --fps 60 --port "$port" --duration 180 --allow-root \
  --acquisition-scenario g6_replay --exposure-ms "$exposure" --exposure-samples 5 \
  --g6-scene-usd /g1/robot_v6.usda --g6-trajectory "$frozen" --g6-trajectory-name "$trajectory" \
  --g6-mount-config /poc/config/zedx_wide.json --g6-head-pose "$head_pose" \
  --g6-receiver-ready "/poc/runs/$receiver/metadata.json" > "$poc/runs/$sender.log" 2>&1 &
sender_pid=$!
finish_sender() {
  [[ -n "$sender_pid" ]] || return
  [[ ! -d "$poc/runs/$sender" ]] || touch "$poc/runs/$sender/stop"
  for _ in {1..40}; do kill -0 "$sender_pid" 2>/dev/null || break; sleep .5; done
  if kill -0 "$sender_pid" 2>/dev/null; then kill -TERM "$sender_pid"; fi
  for _ in {1..20}; do kill -0 "$sender_pid" 2>/dev/null || break; sleep .5; done
  if kill -0 "$sender_pid" 2>/dev/null; then kill -KILL "$sender_pid"; fi
  wait "$sender_pid"; sender_result=$?
  printf '%s\n' "$sender_result" > "$poc/runs/$sender.exitcode"
  sender_pid=''
}
trap finish_sender EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS+600))
while [[ ! -f "$poc/runs/$sender/ready" ]]; do
  if ! kill -0 "$sender_pid" 2>/dev/null || ((SECONDS>=deadline)); then exit 1; fi
  sleep .5
done
frames=$(python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));print(len(m["target_visual_ids"])+m["acquisition"]["preroll_frames"])' "$poc/runs/$sender/metadata.json")
sha256sum "$poc/receiver_build/zed_sdk_receiver" > "$poc/runs/$receiver.binary_sha256.txt"
env ZED_SDK_PATH=/poc/sdk-neural \
  LD_LIBRARY_PATH=/mnt/sdk-5.1.0-unpacked/lib:/mnt/sdk-5.1.0-unpacked/dependencies/ai_dependencies/lib \
  CUDA_VISIBLE_DEVICES=GPU-3c0a37f5-27f6-5b9e-64a4-7c59aadf0cd2 \
  bash "$poc/run_g6a_in_ubuntu22.sh" /poc/receiver_build/zed_sdk_receiver \
  --ip 127.0.0.1 --port "$port" --frames "$frames" --output "/poc/runs/$receiver" --gpu 0 \
  --depth-mode NEURAL --timeout-seconds 180 --calibration-file /mnt/calibration/SN45626933.conf \
  --calibration-sha256 a4856611b40913db85d6968430c4d0a80e296c01607ce655950f313618cafb7e \
  > "$poc/runs/$receiver.log" 2>&1
receiver_result=$?
printf '%s\n' "$receiver_result" > "$poc/runs/$receiver.exitcode"
finish_sender
echo "sender=$sender_result receiver=$receiver_result"
[[ "$sender_result" == 0 && "$receiver_result" == 0 ]]
