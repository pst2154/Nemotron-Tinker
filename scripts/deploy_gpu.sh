#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

ACTION="${1:-start}"

GPU_HOST="${GPU_HOST:-}"
REMOTE_HOME_SCRATCH="${REMOTE_HOME_SCRATCH:-/home/scratch.${USER}}"
REMOTE_ROOT="${REMOTE_ROOT:-${REMOTE_HOME_SCRATCH}/nemotron-tinker-deploy}"
REMOTE_SCRATCH="${REMOTE_SCRATCH:-${REMOTE_HOME_SCRATCH}/nemotron_tinker_ui}"
AUTOMODEL_DIR="${AUTOMODEL_DIR:-${REMOTE_ROOT}/Automodel}"
TINKER_DIR="${TINKER_DIR:-${REMOTE_ROOT}/Nemotron-Tinker}"
BASE_MODEL="${BASE_MODEL:-${REMOTE_HOME_SCRATCH}/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${REMOTE_HOME_SCRATCH}/hf}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-tinker-ui}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"
REMOTE_PORT="${REMOTE_PORT:-18080}"
LOCAL_PORT="${LOCAL_PORT:-18081}"
START_TUNNEL="${START_TUNNEL:-1}"
ENABLE_EXTERNAL_RL_WORKER="${ENABLE_EXTERNAL_RL_WORKER:-0}"
RUN_AS_HOST_USER="${RUN_AS_HOST_USER:-0}"

ssh_remote() {
  if [ -z "${GPU_HOST}" ]; then
    echo "Set GPU_HOST to the target SSH host." >&2
    exit 2
  fi
  ssh "${GPU_HOST}" "$@"
}

print_config() {
  cat <<EOF
Nemotron-Tinker deployment
  host:              ${GPU_HOST}
  repo:              ${TINKER_DIR}
  automodel:         ${AUTOMODEL_DIR}
  scratch:           ${REMOTE_SCRATCH}
  base model:        ${BASE_MODEL}
  image:             ${CONTAINER_IMAGE}
  container:         ${CONTAINER_NAME}
  remote port:       ${REMOTE_PORT}
  local UI:          http://127.0.0.1:${LOCAL_PORT}/ui
  external RL worker ${ENABLE_EXTERNAL_RL_WORKER}
  run as host user:  ${RUN_AS_HOST_USER}
EOF
}

start_remote() {
  print_config
  ssh_remote \
    "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_SCRATCH='${REMOTE_SCRATCH}' REMOTE_HOME_SCRATCH='${REMOTE_HOME_SCRATCH}' AUTOMODEL_DIR='${AUTOMODEL_DIR}' TINKER_DIR='${TINKER_DIR}' BASE_MODEL='${BASE_MODEL}' HF_CACHE_DIR='${HF_CACHE_DIR}' CONTAINER_IMAGE='${CONTAINER_IMAGE}' CONTAINER_NAME='${CONTAINER_NAME}' CUDA_VISIBLE_DEVICES_VALUE='${CUDA_VISIBLE_DEVICES_VALUE}' REMOTE_PORT='${REMOTE_PORT}' ENABLE_EXTERNAL_RL_WORKER='${ENABLE_EXTERNAL_RL_WORKER}' RUN_AS_HOST_USER='${RUN_AS_HOST_USER}' bash -s" <<'REMOTE'
set -euo pipefail

if [ ! -d "${TINKER_DIR}/.git" ]; then
  echo "Missing Nemotron-Tinker checkout: ${TINKER_DIR}" >&2
  echo "Clone it there first, or set TINKER_DIR." >&2
  exit 2
fi
if [ ! -d "${AUTOMODEL_DIR}" ]; then
  echo "Missing AutoModel checkout: ${AUTOMODEL_DIR}" >&2
  echo "Clone it there first, or set AUTOMODEL_DIR." >&2
  exit 2
fi
if [ ! -e "${BASE_MODEL}" ]; then
  echo "Missing base model path: ${BASE_MODEL}" >&2
  exit 2
fi

cd "${TINKER_DIR}"
git pull --ff-only origin main

mkdir -p "${REMOTE_SCRATCH}" "${HF_CACHE_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

user_args=()
if [ "${RUN_AS_HOST_USER}" = "1" ]; then
  mkdir -p "${REMOTE_SCRATCH}/cache/torchinductor" "${REMOTE_SCRATCH}/cache/xdg" "${REMOTE_SCRATCH}/cache/tmp"
  user_args=(
    --user "$(id -u):$(id -g)"
    -e "USER=${USER:-nemotron}"
    -e "LOGNAME=${USER:-nemotron}"
    -e "HOME=${REMOTE_HOME_SCRATCH}"
    -e "XDG_CACHE_HOME=${REMOTE_SCRATCH}/cache/xdg"
    -e "TORCHINDUCTOR_CACHE_DIR=${REMOTE_SCRATCH}/cache/torchinductor"
    -e "TMPDIR=${REMOTE_SCRATCH}/cache/tmp"
  )
fi

docker run -d --name "${CONTAINER_NAME}" \
  "${user_args[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  -e PYTHONPATH=/workspace/Nemotron-Tinker/src:/workspace/Nemotron-Tinker:/workspace/Automodel \
  -v "${REMOTE_HOME_SCRATCH}:${REMOTE_HOME_SCRATCH}" \
  -v "${AUTOMODEL_DIR}:/workspace/Automodel" \
  -v "${TINKER_DIR}:/workspace/Nemotron-Tinker" \
  -w /workspace/Nemotron-Tinker \
  "${CONTAINER_IMAGE}" \
  python scripts/run_mixed_lora_server.py \
    --base-model "${BASE_MODEL}" \
    --scratch-dir "${REMOTE_SCRATCH}" \
    --cache-dir "${HF_CACHE_DIR}" \
    --rank 8 \
    --alpha 16 \
    --mixed-lora-backend grouped \
    --attn-implementation eager \
    --torch-dtype bfloat16 \
    --trust-remote-code \
    --target-modules q_proj k_proj v_proj o_proj \
    --restore-runs-on-startup \
    --host 127.0.0.1 \
    --port "${REMOTE_PORT}" >/dev/null

if [ "${ENABLE_EXTERNAL_RL_WORKER}" = "1" ]; then
  worker_state="${REMOTE_SCRATCH}_host_rl_launcher"
  mkdir -p "${worker_state}"
  pkill -f "host_rl_launcher.py --scratch-dir ${REMOTE_SCRATCH}" >/dev/null 2>&1 || true
  nohup python3 "${TINKER_DIR}/tools/host_rl_launcher.py" \
    --scratch-dir "${REMOTE_SCRATCH}" \
    --poll-seconds 2 \
    --api-url "http://127.0.0.1:${REMOTE_PORT}" \
    > "${worker_state}/worker.out" 2>&1 &
fi

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${REMOTE_PORT}/health" >/tmp/nemotron_tinker_health.json 2>/dev/null; then
    cat /tmp/nemotron_tinker_health.json
    echo
    exit 0
  fi
  sleep 5
done

echo "Service did not become healthy. Last container logs:" >&2
docker logs --tail 160 "${CONTAINER_NAME}" >&2 || true
exit 1
REMOTE
}

start_tunnel() {
  if [ "${START_TUNNEL}" != "1" ]; then
    return
  fi
  if [ -z "${GPU_HOST}" ]; then
    echo "Set GPU_HOST to the target SSH host." >&2
    exit 2
  fi
  if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Local port ${LOCAL_PORT} is already listening; leaving it alone."
  else
    ssh -f -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" -o ExitOnForwardFailure=yes "${GPU_HOST}"
  fi
  echo "UI: http://127.0.0.1:${LOCAL_PORT}/ui"
}

status_remote() {
  ssh_remote "docker ps --filter name='${CONTAINER_NAME}' --format '{{.Names}} {{.Image}} {{.Status}}'; curl -fsS http://127.0.0.1:${REMOTE_PORT}/health"
}

stop_remote() {
  ssh_remote "docker rm -f '${CONTAINER_NAME}' >/dev/null 2>&1 || true; pkill -f 'host_rl_launcher.py --scratch-dir ${REMOTE_SCRATCH}' >/dev/null 2>&1 || true"
}

case "${ACTION}" in
  start)
    start_remote
    start_tunnel
    ;;
  tunnel)
    start_tunnel
    ;;
  status)
    status_remote
    ;;
  stop)
    stop_remote
    ;;
  *)
    echo "Usage: $0 [start|tunnel|status|stop]" >&2
    exit 2
    ;;
esac
