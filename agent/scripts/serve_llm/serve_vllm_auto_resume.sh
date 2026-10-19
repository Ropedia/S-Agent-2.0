#!/bin/bash

set -u

TARGET_SCRIPT="/data/code/serve_llm/serve_vllm_glm5.1_fp8.sh"
RESTART_DELAY_SECONDS=5
GPU_MEM_THRESHOLD_PERCENT=10
GPU_CHECK_INTERVAL_SECONDS=10

child_pid=""
stop_requested=0

timestamp() {
	date '+%Y-%m-%d %H:%M:%S'
}

log() {
	echo "[$(timestamp)] $*"
}

gpu_mem_below_threshold() {
	local gpu_line
	local used_mem
	local total_mem
	local saw_gpu=0

	if ! command -v nvidia-smi >/dev/null 2>&1; then
		log "nvidia-smi not found. Cannot verify GPU memory usage."
		return 1
	fi

	while IFS=',' read -r used_mem total_mem; do
		saw_gpu=1
		used_mem="${used_mem//[[:space:]]/}"
		total_mem="${total_mem//[[:space:]]/}"

		if [[ -z "${used_mem}" || -z "${total_mem}" || "${total_mem}" -eq 0 ]]; then
			log "Invalid GPU memory stats from nvidia-smi."
			return 1
		fi

		if (( used_mem * 100 >= total_mem * GPU_MEM_THRESHOLD_PERCENT )); then
			return 1
		fi
	done < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

	if [[ "${saw_gpu}" -eq 0 ]]; then
		log "No GPU memory stats returned from nvidia-smi."
		return 1
	fi

	return 0
}

wait_for_gpu_mem_ready() {
	while true; do
		if gpu_mem_below_threshold; then
			log "All GPU memory usage is below ${GPU_MEM_THRESHOLD_PERCENT}%."
			return 0
		fi

		if [[ "${stop_requested}" -eq 1 ]]; then
			return 1
		fi

		log "GPU memory usage is not below ${GPU_MEM_THRESHOLD_PERCENT}%. Retrying in ${GPU_CHECK_INTERVAL_SECONDS}s."
		sleep "${GPU_CHECK_INTERVAL_SECONDS}"
	done
}

stop_monitor() {
	stop_requested=1
	log "Stop signal received. Stopping monitor."

	if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
		kill -TERM "${child_pid}" 2>/dev/null || true
	fi
}

trap stop_monitor INT TERM

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
	log "Target script not found: ${TARGET_SCRIPT}"
	exit 1
fi

log "Monitoring target script: ${TARGET_SCRIPT}"
log "Restart delay: ${RESTART_DELAY_SECONDS}s"

while true; do
	if ! wait_for_gpu_mem_ready; then
		log "Stop requested before target script start. Exiting monitor."
		break
	fi

	log "Starting target script."
	bash "${TARGET_SCRIPT}" &
	child_pid=$!

	wait "${child_pid}"
	exit_code=$?
	child_pid=""

	if [[ "${stop_requested}" -eq 1 ]]; then
		log "Target script stopped. Exiting monitor."
		break
	fi

	log "Target script exited with code ${exit_code}. Restarting in ${RESTART_DELAY_SECONDS}s."
	sleep "${RESTART_DELAY_SECONDS}"
done
