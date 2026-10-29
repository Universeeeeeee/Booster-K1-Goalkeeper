#!/usr/bin/env bash
# Target-specific isolated G5 trial; dependencies remain read-only.
set -uo pipefail
poc=/home/lizhengran/isaacsim-workspace/g5_static
deps=/home/lizhengran/isaacsim-workspace/g2_zed_poc
source_gpu_uuid=${G5_SOURCE_GPU_UUID:-GPU-50273ff3-fae7-f61c-4197-088b4d8a0037}
render_gpu_index=${G5_RENDER_GPU_INDEX:-0}
receiver_gpu_uuid=${G5_RECEIVER_GPU_UUID:-GPU-3c0a37f5-27f6-5b9e-64a4-7c59aadf0cd2}
[[ $# -ge 3 ]] || { echo "Usage: sender receiver scenario [SDK frames] [exposure ms] [source frames] [samples] [port] [ipc|network]" >&2; exit 2; }
sender=$1
receiver=$2
scenario=$3
frames=${4:-185}
exposure_ms=${5:-0}
source_frames=${6:-204}
samples=${7:-3}
port=${8:-30000}
transport=${9:-ipc}
receiver_binary=${G5_RECEIVER_BINARY:-/mnt/receiver_build_51/zed_sdk_receiver}
receiver_options=()
if [[ -n "${G5_DEPTH_MODE:-}" ]]; then receiver_options=(--depth-mode "$G5_DEPTH_MODE"); fi
if [[ -n "${G5_RECORD_DISPARITY:-}" ]]; then receiver_options+=(--record-disparity "$G5_RECORD_DISPARITY"); fi
[[ "$transport" == ipc || "$transport" == network ]] || exit 2
fps=60
[[ "$scenario" != baseline ]] || fps=30
ready_frames=1
[[ "$scenario" != baseline ]] || ready_frames=10
for name in "$sender" "$receiver"; do
  [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run name" >&2; exit 2; }
  for suffix in "" .log .exitcode; do
    [[ ! -e "$poc/runs/$name$suffix" ]] || { echo "Run already exists: $name$suffix" >&2; exit 2; }
  done
done
[[ "$sender" != "$receiver" ]] || exit 2
extra=()
if [[ "$scenario" != baseline ]]; then extra=(--acquisition-scenario "$scenario" --exposure-ms "$exposure_ms" --exposure-samples "$samples" --acquisition-frames "$source_frames"); fi
if [[ -n "${G5_TEXTURE:-}" ]]; then extra+=(--g5-texture "$G5_TEXTURE"); fi
if [[ "$transport" == network ]]; then extra+=(--network); fi
env CUDA_VISIBLE_DEVICES="$source_gpu_uuid" \
  PYTHONPATH=/poc:/opt/python_deps PYTHONUNBUFFERED=1 \
  bash "$poc/run_g5_in_ubuntu22.sh" /isaac/python.sh /poc/zed_sim_poc.py \
  --extension-root /opt/zed-v3.2.0-combined/exts \
  --camera-usd /opt/zed-v3.2.0-combined/usd/ZED_X.usdc \
  --output "/poc/runs/$sender" --gpu 0 --render-gpu-index "$render_gpu_index" --fps "$fps" --port "$port" --duration 180 --allow-root \
  "${extra[@]}" > "$poc/runs/$sender.log" 2>&1 &
sender_pid=$!
finish_sender() {
  [[ -n "$sender_pid" ]] || return
  [[ ! -d "$poc/runs/$sender" ]] || touch "$poc/runs/$sender/stop"
  for _ in {1..20}; do
    kill -0 "$sender_pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$sender_pid" 2>/dev/null; then
    kill -TERM "$sender_pid"
    for _ in {1..20}; do
      kill -0 "$sender_pid" 2>/dev/null || break
      sleep 0.5
    done
  fi
  if kill -0 "$sender_pid" 2>/dev/null; then kill -KILL "$sender_pid"; fi
  wait "$sender_pid"; sender_result=$?
  printf '%s\n' "$sender_result" > "$poc/runs/$sender.exitcode"
  sender_pid=""
}
trap finish_sender EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 600))
while [[ ! -f "$poc/runs/$sender/ready" ]] || \
      [[ $(wc -l < "$poc/runs/$sender/source_frames.jsonl" 2>/dev/null || echo 0) -lt "$ready_frames" ]]; do
  if ! kill -0 "$sender_pid" 2>/dev/null; then
    finish_sender
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    kill -TERM "$sender_pid"
    finish_sender
    exit 1
  fi
  sleep 0.5
done
env ZED_SDK_PATH="${G5_SDK_PATH:-/mnt/sdk-5.1.0-unpacked}" \
  LD_LIBRARY_PATH=/mnt/sdk-5.1.0-unpacked/lib:/mnt/sdk-5.1.0-unpacked/dependencies/ai_dependencies/lib \
  CUDA_VISIBLE_DEVICES="$receiver_gpu_uuid" \
  bash "$poc/run_g5_in_ubuntu22.sh" "$receiver_binary" \
  --ip 127.0.0.1 --port "$port" --frames "$frames" \
  --output "/poc/runs/$receiver" --gpu 0 --timeout-seconds "${G5_RECEIVER_TIMEOUT_SECONDS:-120}" \
  --calibration-file /mnt/calibration/SN45626933.conf \
  --calibration-sha256 a4856611b40913db85d6968430c4d0a80e296c01607ce655950f313618cafb7e \
  "${receiver_options[@]}" \
  > "$poc/runs/$receiver.log" 2>&1
receiver_result=$?
printf '%s\n' "$receiver_result" > "$poc/runs/$receiver.exitcode"
finish_sender
echo "sender=$sender_result receiver=$receiver_result"
[[ "$sender_result" == 0 && "$receiver_result" == 0 ]]
