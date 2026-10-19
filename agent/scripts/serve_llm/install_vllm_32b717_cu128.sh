#!/bin/bash

set -euo pipefail

# Install vLLM from commit 32b7177909d1c9928bcedd81de7de5a1fa21d2b3
# plus the DeepGEMM build required by GLM-5.1-FP8.
#
# Usage:
#   bash /data/code/serve_llm/install_vllm_32b717_cu128.sh
#
# Optional overrides:
#   CONDA_ENV_NAME=vllm-32b717
#   PYTHON_VERSION=3.12
#   CUDA_HOME=/usr/local/cuda
#   VLLM_SRC_DIR=/data/code/vllm-src/vllm-32b717
#   VLLM_COMMIT=32b7177909d1c9928bcedd81de7de5a1fa21d2b3
#   TORCH_CUDA_ARCH_LIST=9.0a
#   INSTALL_DEEPGEMM=1
#   DEEPGEMM_GIT_REF=891d57b4db1071624b5c8fa0d1e51cb317fa709f
#   DEEPGEMM_SRC_DIR=/data/code/deepgemm-glm5.1-fp8
#   WHEEL_DIR=/data/packages/deepgemm
#   FORCE_RECLONE=1

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vllm-32b717}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VLLM_COMMIT="${VLLM_COMMIT:-32b7177909d1c9928bcedd81de7de5a1fa21d2b3}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-/data/code/vllm-src/vllm-32b717}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
DEEPGEMM_GIT_REPO="${DEEPGEMM_GIT_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
# Keep this at the GLM-5.1-FP8-compatible DeepGEMM ref that contains
# fp8_fp4_paged_mqa_logits.
DEEPGEMM_GIT_REF="${DEEPGEMM_GIT_REF:-891d57b4db1071624b5c8fa0d1e51cb317fa709f}"
DEEPGEMM_SRC_DIR="${DEEPGEMM_SRC_DIR:-/data/code/deepgemm-glm5.1-fp8}"
WHEEL_DIR="${WHEEL_DIR:-}"

timestamp() {
	date '+%Y-%m-%d %H:%M:%S'
}

log() {
	echo "[$(timestamp)] $*"
}

require_command() {
	local command_name="$1"

	if ! command -v "${command_name}" >/dev/null 2>&1; then
		echo "Missing required command: ${command_name}" >&2
		exit 1
	fi
}

load_conda() {
	if command -v conda >/dev/null 2>&1; then
		local conda_base
		conda_base="$(conda info --base)"
		# shellcheck disable=SC1091
		source "${conda_base}/etc/profile.d/conda.sh"
		return
	fi

	for conda_sh in \
		"/data/miniconda3/etc/profile.d/conda.sh" \
		"${HOME}/miniconda3/etc/profile.d/conda.sh" \
		"${HOME}/miniconda/etc/profile.d/conda.sh" \
		"/mnt/djb/miniconda/etc/profile.d/conda.sh" \
		"/opt/conda/etc/profile.d/conda.sh"; do
		if [[ -f "${conda_sh}" ]]; then
			# shellcheck disable=SC1090
			source "${conda_sh}"
			return
		fi
	done

	echo "Could not find conda. Please install conda or set PATH to include it." >&2
	exit 1
}

detect_cuda_version_from_nvcc() {
	local nvcc_path="$1"
	"${nvcc_path}" --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1
}

check_cuda() {
	if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
		echo "nvcc not found at ${CUDA_HOME}/bin/nvcc." >&2
		echo "Set CUDA_HOME=/path/to/cuda and rerun." >&2
		exit 1
	fi

	log "Using CUDA_HOME=${CUDA_HOME}"
	export CUDA_HOME
	export CUDA_PATH="${CUDA_HOME}"
	export TORCH_CUDA_ARCH_LIST
	export PATH="${CUDA_HOME}/bin:${PATH}"
	export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

	local cuda_version
	cuda_version="$(detect_cuda_version_from_nvcc "${CUDA_HOME}/bin/nvcc")"
	if [[ -z "${cuda_version}" ]]; then
		echo "Could not detect CUDA version from ${CUDA_HOME}/bin/nvcc." >&2
		exit 1
	fi

	local cuda_major="${cuda_version%%.*}"
	local cuda_minor="${cuda_version#${cuda_major}.}"
	cuda_minor="${cuda_minor%%.*}"
	if (( cuda_major < 12 || (cuda_major == 12 && cuda_minor < 8) )); then
		echo "CUDA 12.8+ is required for cu128/GLM-5.1-FP8, got ${cuda_version}." >&2
		exit 1
	fi

	log "Detected CUDA ${cuda_version}"
	log "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
}

create_or_activate_env() {
	if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
		log "Creating conda env: ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})"
		conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
	else
		log "Conda env already exists: ${CONDA_ENV_NAME}"
	fi

	conda activate "${CONDA_ENV_NAME}"
}

install_uv() {
	log "Installing uv"
	python -m pip install -U pip uv
}

fetch_vllm_source() {
	require_command git

	if [[ -d "${VLLM_SRC_DIR}/.git" ]]; then
		log "Updating existing vLLM source: ${VLLM_SRC_DIR}"
		git -C "${VLLM_SRC_DIR}" fetch origin
	else
		log "Cloning vLLM into ${VLLM_SRC_DIR}"
		mkdir -p "$(dirname "${VLLM_SRC_DIR}")"
		git clone https://github.com/vllm-project/vllm.git "${VLLM_SRC_DIR}"
	fi

	log "Checking out vLLM commit: ${VLLM_COMMIT}"
	git -C "${VLLM_SRC_DIR}" checkout "${VLLM_COMMIT}"
}

patch_vllm_for_cuda128_flashmla() {
	cd "${VLLM_SRC_DIR}"

	log "Patching vLLM FlashMLA build gate for CUDA 12.8"
	python - <<'PY'
from pathlib import Path

setup_py = Path("setup.py")
text = setup_py.read_text()
old = """    if envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9")
    ):
        # FlashMLA requires CUDA 12.9 or later
"""
new = """    if envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.8")
    ):
        # FlashMLA is required by GLM-5.1-FP8 and builds for Hopper on CUDA 12.8.
"""
if old not in text:
    raise SystemExit("Could not find the expected FlashMLA CUDA version gate in setup.py")
setup_py.write_text(text.replace(old, new))
PY
}

install_vllm() {
	cd "${VLLM_SRC_DIR}"

	log "Preparing vLLM to use the existing PyTorch installation"
	python use_existing_torch.py

	log "Installing vLLM build requirements for cu128"
	uv pip install -r requirements/build/cuda.txt --torch-backend=cu128

	log "Cleaning previous vLLM build artifacts"
	rm -rf build dist *.egg-info

	log "Building and installing vLLM editable package with FlashMLA support"
	uv pip install --no-build-isolation -e . --torch-backend=cu128
}

fetch_deepgemm_source() {
	if [[ "${INSTALL_DEEPGEMM}" != "1" ]]; then
		log "Skipping DeepGEMM because INSTALL_DEEPGEMM=${INSTALL_DEEPGEMM}"
		return
	fi

	require_command git

	if [[ "${FORCE_RECLONE:-0}" == "1" ]]; then
		log "Removing existing DeepGEMM source: ${DEEPGEMM_SRC_DIR}"
		rm -rf "${DEEPGEMM_SRC_DIR}"
	fi

	if [[ -d "${DEEPGEMM_SRC_DIR}/.git" ]]; then
		log "Updating existing DeepGEMM source: ${DEEPGEMM_SRC_DIR}"
		git -C "${DEEPGEMM_SRC_DIR}" fetch origin
	else
		log "Cloning DeepGEMM into ${DEEPGEMM_SRC_DIR}"
		mkdir -p "$(dirname "${DEEPGEMM_SRC_DIR}")"
		git clone --recursive --shallow-submodules "${DEEPGEMM_GIT_REPO}" "${DEEPGEMM_SRC_DIR}"
	fi

	log "Checking out DeepGEMM ref: ${DEEPGEMM_GIT_REF}"
	git -C "${DEEPGEMM_SRC_DIR}" checkout "${DEEPGEMM_GIT_REF}"
	git -C "${DEEPGEMM_SRC_DIR}" submodule update --init --recursive
}

build_and_install_deepgemm() {
	if [[ "${INSTALL_DEEPGEMM}" != "1" ]]; then
		return
	fi

	cd "${DEEPGEMM_SRC_DIR}"

	log "Removing previous DeepGEMM build artifacts"
	rm -rf build dist *.egg-info

	log "Installing DeepGEMM build dependencies"
	python -m pip install -U pip wheel setuptools

	log "Building DeepGEMM wheel for GLM-5.1-FP8"
	python setup.py bdist_wheel

	if [[ -n "${WHEEL_DIR}" ]]; then
		log "Copying DeepGEMM wheel to ${WHEEL_DIR}"
		mkdir -p "${WHEEL_DIR}"
		cp dist/*.whl "${WHEEL_DIR}/"
	fi

	log "Uninstalling previous deep_gemm package, if present"
	python -m pip uninstall -y deep_gemm || true

	log "Installing newly built DeepGEMM wheel"
	python -m pip install --force-reinstall dist/*.whl
}

verify_install() {
	log "Verifying installation"
	python - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
PY

	if [[ "${INSTALL_DEEPGEMM}" == "1" ]]; then
		log "Verifying DeepGEMM symbols required by GLM-5.1-FP8"
		(
			cd /tmp
			python - <<'PY'
import deep_gemm

required = [
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
]

print("deep_gemm file:", getattr(deep_gemm, "__file__", None))
print("deep_gemm version:", getattr(deep_gemm, "__version__", None))

missing = [name for name in required if not hasattr(deep_gemm, name)]
if missing:
    raise SystemExit("Missing required DeepGEMM symbols: " + ", ".join(missing))

print("DeepGEMM GLM-5.1-FP8 symbol check: OK")
PY
		)
	fi

	log "Verifying vLLM FlashMLA extensions required by GLM-5.1-FP8"
	python - <<'PY'
import importlib.util
import torch
import vllm

required = ["vllm._flashmla_C", "vllm._flashmla_extension_C"]
print("vllm file:", getattr(vllm, "__file__", None))
print("vllm version:", getattr(vllm, "__version__", None))
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu capability:", torch.cuda.get_device_capability(0))
    print("gpu name:", torch.cuda.get_device_name(0))

missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required vLLM FlashMLA extensions: " + ", ".join(missing))

print("vLLM FlashMLA extension check: OK")
PY
}

main() {
	load_conda
	check_cuda
	create_or_activate_env
	install_uv
	fetch_vllm_source
	patch_vllm_for_cuda128_flashmla
	install_vllm
	fetch_deepgemm_source
	build_and_install_deepgemm
	verify_install

	log "Done. Activate with: conda activate ${CONDA_ENV_NAME}"
	log "For GLM-5.1-FP8, use this env in your serve script before running vllm serve."
}

main "$@"
