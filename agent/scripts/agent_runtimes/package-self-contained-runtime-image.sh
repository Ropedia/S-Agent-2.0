#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Package a self-contained custom agent runtime as a Docker image.

This mode installs a uv-managed standalone Python inside the runtime directory,
installs the required packages there, and creates a wrapper bin/python for the
mounted runtime.

Usage:
  bash scripts/agent_runtimes/package-self-contained-runtime-image.sh \
    --image docker.io/yourname/c-oh-sdk-1.33.0:v1 \
  --python-version 3.12 \
  --uv-version 0.5.21 \
  --runtime-source-dir scripts/agent_runtimes/openhands-sdk/1.33.0/runtime \
  --container-runtime-root /opt/custom-agent-runtime/oh-sdk \
  --agent openhands-sdk \
  --runtime-version 1.33.0 \
  --packages-json '{"openhands-sdk":"1.33.0","openhands-tools":"1.33.0","fastapi":"0.115.6","platformdirs":"4.3.6"}'

Options:
  --image NAME                Target image tag.
  --python-version VERSION    Python major.minor version to embed.
  --python-abis LIST          Comma-separated standalone Python ABIs to embed.
                              Supported: glibc, musl. Default: glibc
  --uv-version VERSION        Pin for the uv tool installed in the image. Default: 0.5.21
  --runtime-source-dir PATH   Directory containing runtime entrypoint files.
  --container-runtime-root    Runtime path inside the image.
  --agent NAME                Agent name for metadata labels.
  --runtime-version VERSION   Runtime version for metadata labels.
  --packages-json JSON        JSON object of package names to versions.
  --base-image IMAGE          Base image used to seed the embedded Python home.
                              Default: python:<version>-slim
  --push                      Push the image after building it.
  -h, --help                  Show this help message.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_PYTHON=(uv run --project "${REPO_ROOT}" python)

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

IMAGE_NAME=""
PYTHON_VERSION=""
PYTHON_ABIS="glibc"
UV_VERSION="0.5.21"
RUNTIME_SOURCE_DIR=""
CONTAINER_RUNTIME_ROOT="/opt/custom-agent-runtime/oh-sdk"
AGENT_NAME=""
RUNTIME_VERSION=""
PACKAGES_JSON=""
BASE_IMAGE=""
PUSH_IMAGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_NAME="${2:-}"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --python-abis)
      PYTHON_ABIS="${2:-}"
      shift 2
      ;;
    --uv-version)
      UV_VERSION="${2:-}"
      shift 2
      ;;
    --runtime-source-dir)
      RUNTIME_SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --container-runtime-root)
      CONTAINER_RUNTIME_ROOT="${2:-}"
      shift 2
      ;;
    --agent)
      AGENT_NAME="${2:-}"
      shift 2
      ;;
    --runtime-version)
      RUNTIME_VERSION="${2:-}"
      shift 2
      ;;
    --packages-json)
      PACKAGES_JSON="${2:-}"
      shift 2
      ;;
    --base-image)
      BASE_IMAGE="${2:-}"
      shift 2
      ;;
    --push)
      PUSH_IMAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${IMAGE_NAME}" || -z "${PYTHON_VERSION}" || -z "${RUNTIME_SOURCE_DIR}" || -z "${PACKAGES_JSON}" ]]; then
  echo "--image, --python-version, --runtime-source-dir, and --packages-json are required." >&2
  usage >&2
  exit 1
fi

require_command docker
require_command uv
require_command realpath

RUNTIME_SOURCE_DIR="$(realpath "${RUNTIME_SOURCE_DIR}")"
if [[ ! -d "${RUNTIME_SOURCE_DIR}" ]]; then
  echo "Runtime source directory not found: ${RUNTIME_SOURCE_DIR}" >&2
  exit 1
fi

PACKAGES_JSON="$(PACKAGES_JSON="${PACKAGES_JSON}" "${HOST_PYTHON[@]}" - <<'PY'
import json
import os

packages = json.loads(os.environ["PACKAGES_JSON"])
print(json.dumps(packages, separators=(",", ":"), sort_keys=True))
PY
)"

PYTHON_ABIS="$(
  PYTHON_ABIS="${PYTHON_ABIS}" "${HOST_PYTHON[@]}" - <<'PY'
import os
import sys

raw_abis = [
    part.strip()
    for part in os.environ["PYTHON_ABIS"].replace(",", " ").split()
    if part.strip()
]
if not raw_abis:
    print("At least one Python ABI must be provided.", file=sys.stderr)
    raise SystemExit(1)

supported = {"glibc", "musl"}
seen = set()
abis = []
for abi in raw_abis:
    if abi not in supported:
        print(
            f"Unsupported Python ABI: {abi}. Supported values: glibc, musl",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if abi not in seen:
        seen.add(abi)
        abis.append(abi)

print(" ".join(abis))
PY
)"

if [[ -z "${BASE_IMAGE}" ]]; then
  BASE_IMAGE="python:${PYTHON_VERSION}-slim-bookworm"
fi

TMP_DIR="$(mktemp -d)"
BUILD_CONTEXT="${TMP_DIR}/context"
RUNTIME_CONTEXT_DIR="${BUILD_CONTEXT}/runtime-source"
METADATA_SCRIPT_PATH="${BUILD_CONTEXT}/write-runtime-metadata.py"
WRAPPER_SCRIPT_PATH="${BUILD_CONTEXT}/python-wrapper.sh"
PIP_WRAPPER_SCRIPT_PATH="${BUILD_CONTEXT}/pip-wrapper.sh"
RUNTIME_SELECT_PYTHON_PATH="${BUILD_CONTEXT}/runtime-select-python.sh"
RUNTIME_ENV_PATH="${BUILD_CONTEXT}/runtime-env.sh"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${RUNTIME_CONTEXT_DIR}"
cp -a "${RUNTIME_SOURCE_DIR}/." "${RUNTIME_CONTEXT_DIR}/"

PACKAGE_SPECS="$(
  PACKAGES_JSON="${PACKAGES_JSON}" "${HOST_PYTHON[@]}" - <<'PY'
import json
import os

packages = json.loads(os.environ["PACKAGES_JSON"])
specs = []
for name, version in packages.items():
    if version == "latest":
        specs.append(name)
    else:
        specs.append(f"{name}=={version}")
print(" ".join(specs))
PY
)"

PACKAGE_SPECS_ESCAPED="$(printf '%s' "${PACKAGE_SPECS}" | sed 's/[[:space:]]\+/ /g')"

MUSL_PYTHON_STAGE=""
MUSL_PYTHON_COPY=""
if [[ " ${PYTHON_ABIS} " == *" musl "* ]]; then
  read -r -d '' MUSL_PYTHON_STAGE <<EOF || true
FROM python:${PYTHON_VERSION}-alpine AS musl-python-builder

RUN apk add --no-cache ca-certificates libgcc libstdc++
RUN python -m pip install --no-cache-dir uv==${UV_VERSION}

ENV UV_LINK_MODE=copy
COPY runtime-source/ /tmp/runtime-source/
RUN uv pip install --python /usr/local/bin/python --break-system-packages ${PACKAGE_SPECS_ESCAPED}
RUN /usr/local/bin/python -c "import openhands.sdk" && \\
    /usr/local/bin/python -m pip show openhands-sdk >/dev/null && \\
    /usr/local/bin/python /tmp/runtime-source/run_openhands.py --help >/dev/null

RUN mkdir -p /opt/musl-python-home/lib && \\
    cp -a /usr/local/. /opt/musl-python-home/ && \\
    for path in /lib/*.so* /usr/lib/*.so*; do \\
      if [ -f "\$path" ]; then cp -aL "\$path" /opt/musl-python-home/lib/; fi; \\
    done

EOF
  MUSL_PYTHON_COPY="COPY --from=musl-python-builder /opt/musl-python-home/ ${CONTAINER_RUNTIME_ROOT}/python-home/musl/"
fi

cat > "${RUNTIME_SELECT_PYTHON_PATH}" <<EOF
# POSIX sh helper shared by runtime-env.sh and bin wrappers.

sagent_runtime_detect_musl() {
  for path in /lib/ld-musl-*.so* /usr/lib/libc.musl-*.so*; do
    if [ -f "\$path" ]; then
      return 0
    fi
  done

  if command -v ldd >/dev/null 2>&1; then
    case "\$(ldd --version 2>&1 || true)" in
      *musl*|*Musl*) return 0 ;;
    esac
  fi

  return 1
}

sagent_runtime_detect_old_glibc() {
  if sagent_runtime_detect_musl; then
    return 1
  fi

  version=""
  if command -v getconf >/dev/null 2>&1; then
    glibc_line=\$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
    case "\$glibc_line" in
      glibc\ *) version=\${glibc_line#glibc } ;;
    esac
  fi

  if [ -z "\$version" ] && command -v ldd >/dev/null 2>&1; then
    version=\$(ldd --version 2>&1 | sed -n '1s/.* //p' || true)
  fi

  major=\${version%%.*}
  rest=\${version#*.}
  minor=\${rest%%[^0-9]*}

  case "\$major" in ''|*[!0-9]*) return 1 ;; esac
  case "\$minor" in ''|*[!0-9]*) return 1 ;; esac

  if [ "\$major" -lt 2 ]; then
    return 0
  fi
  if [ "\$major" -eq 2 ] && [ "\$minor" -lt 33 ]; then
    return 0
  fi

  return 1
}

sagent_runtime_selected_python_abi() {
  case "\${CUSTOM_AGENT_RUNTIME_PYTHON_ABI:-}" in
    glibc|musl)
      printf '%s\n' "\$CUSTOM_AGENT_RUNTIME_PYTHON_ABI"
      return 0
      ;;
    "")
      ;;
    *)
      echo "Unsupported CUSTOM_AGENT_RUNTIME_PYTHON_ABI: \$CUSTOM_AGENT_RUNTIME_PYTHON_ABI" >&2
      return 1
      ;;
  esac

  if sagent_runtime_detect_musl; then
    printf '%s\n' musl
  elif sagent_runtime_detect_old_glibc; then
    printf '%s\n' musl
  else
    printf '%s\n' glibc
  fi
}

sagent_runtime_select_python_home() {
  root=\${CUSTOM_AGENT_RUNTIME_ROOT:-${CONTAINER_RUNTIME_ROOT}}
  abi=\$(sagent_runtime_selected_python_abi) || return 1
  home="\$root/python-home/\$abi"
  SAGENT_RUNTIME_SELECTED_PYTHON_ABI="\$abi"
  export SAGENT_RUNTIME_SELECTED_PYTHON_ABI

  if [ ! -x "\$home/bin/python3" ] && [ ! -x "\$home/bin/python" ]; then
    echo "Selected \$abi Python runtime is missing or not executable: \$home" >&2
    return 1
  fi

  printf '%s\n' "\$home"
}

sagent_runtime_resolve_python() {
  home=\$(sagent_runtime_select_python_home) || return 1

  if [ -x "\$home/bin/python3" ]; then
    printf '%s\n' "\$home/bin/python3"
    return 0
  fi

  printf '%s\n' "\$home/bin/python"
}

sagent_runtime_exec_python() {
  home=\$(sagent_runtime_select_python_home) || return 1
  selected_abi=\$(sagent_runtime_selected_python_abi) || return 1

  # Keep the task workspace cwd (e.g. /testbed) off sys.path so embedded
  # site-packages (requests, etc.) are not shadowed. Equivalent to python -P.
  PYTHONSAFEPATH=\${PYTHONSAFEPATH:-1}
  export PYTHONSAFEPATH

  if [ -x "\$home/bin/python3" ]; then
    python_executable="\$home/bin/python3"
  else
    python_executable="\$home/bin/python"
  fi

  if [ "\$selected_abi" = "musl" ]; then
    for loader in "\$home"/lib/ld-musl-*.so*; do
      if [ -f "\$loader" ]; then
        exec "\$loader" --library-path "\$home/lib" "\$python_executable" "\$@"
      fi
    done
  fi

  exec "\$python_executable" "\$@"
}
EOF

cat > "${WRAPPER_SCRIPT_PATH}" <<'EOF'
#!/bin/sh
set -e

runtime_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
CUSTOM_AGENT_RUNTIME_ROOT=${CUSTOM_AGENT_RUNTIME_ROOT:-$runtime_root}
export CUSTOM_AGENT_RUNTIME_ROOT

. "$runtime_root/runtime-select-python.sh"
sagent_runtime_exec_python "$@"
EOF

cat > "${PIP_WRAPPER_SCRIPT_PATH}" <<'EOF'
#!/bin/sh
set -e

runtime_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
CUSTOM_AGENT_RUNTIME_ROOT=${CUSTOM_AGENT_RUNTIME_ROOT:-$runtime_root}
export CUSTOM_AGENT_RUNTIME_ROOT

. "$runtime_root/runtime-select-python.sh"
sagent_runtime_exec_python -m pip "$@"
EOF

cat > "${RUNTIME_ENV_PATH}" <<EOF
# POSIX sh file intended to be sourced by task containers.

CUSTOM_AGENT_RUNTIME_ROOT=\${CUSTOM_AGENT_RUNTIME_ROOT:-${CONTAINER_RUNTIME_ROOT}}
export CUSTOM_AGENT_RUNTIME_ROOT

if [ ! -f "\$CUSTOM_AGENT_RUNTIME_ROOT/runtime-select-python.sh" ]; then
  echo "Runtime Python selector not found: \$CUSTOM_AGENT_RUNTIME_ROOT/runtime-select-python.sh" >&2
  sagent_runtime_status=1
  return "\$sagent_runtime_status" 2>/dev/null || exit "\$sagent_runtime_status"
fi

. "\$CUSTOM_AGENT_RUNTIME_ROOT/runtime-select-python.sh" || {
  sagent_runtime_status=\$?
  return "\$sagent_runtime_status" 2>/dev/null || exit "\$sagent_runtime_status"
}
EMBEDDED_PYTHON_HOME=\$(sagent_runtime_select_python_home) || {
  sagent_runtime_status=\$?
  return "\$sagent_runtime_status" 2>/dev/null || exit "\$sagent_runtime_status"
}
export EMBEDDED_PYTHON_HOME
export PATH="\$CUSTOM_AGENT_RUNTIME_ROOT/bin:\$EMBEDDED_PYTHON_HOME/bin\${PATH:+:\$PATH}"
EOF

cat > "${METADATA_SCRIPT_PATH}" <<'EOF'
import importlib.metadata as md
import json
import os
from pathlib import Path

runtime_root = Path(os.environ["RUNTIME_ROOT"])
packages = json.loads(os.environ["PACKAGES_JSON"])
python_abis = os.environ["PYTHON_ABIS"].split()
package_versions = {}
for package_name in packages:
    package_versions[package_name] = md.version(package_name)

metadata = {
    "runtime_mode": "self-contained",
    "python_executable": str(runtime_root / "bin" / "python"),
    "pip_executable": str(runtime_root / "bin" / "pip"),
    "python_abis": python_abis,
    "python_homes": {
        abi: str(runtime_root / "python-home" / abi) for abi in python_abis
    },
    "runtime_entrypoint": "runtime/run_openhands.py",
    "runtime_env_script": "runtime-env.sh",
    "packages": package_versions,
}
(runtime_root / "runtime-metadata.json").write_text(json.dumps(metadata, indent=2))
EOF

cat > "${BUILD_CONTEXT}/Dockerfile" <<EOF
${MUSL_PYTHON_STAGE}
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates && \\
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv==${UV_VERSION}

ENV SAGENT_RUNTIME_ROOT=${CONTAINER_RUNTIME_ROOT}
ENV SAGENT_RUNTIME_PYTHON_HOME_ROOT=${CONTAINER_RUNTIME_ROOT}/python-home
ENV SAGENT_RUNTIME_PYTHON_ABIS="${PYTHON_ABIS}"
ENV UV_LINK_MODE=copy

RUN mkdir -p \\
      ${CONTAINER_RUNTIME_ROOT}/bin \\
      ${CONTAINER_RUNTIME_ROOT}/runtime \\
      ${CONTAINER_RUNTIME_ROOT}/python-home
COPY runtime-source/ ${CONTAINER_RUNTIME_ROOT}/runtime/
COPY python-wrapper.sh ${CONTAINER_RUNTIME_ROOT}/bin/python
COPY pip-wrapper.sh ${CONTAINER_RUNTIME_ROOT}/bin/pip
COPY runtime-select-python.sh ${CONTAINER_RUNTIME_ROOT}/runtime-select-python.sh
COPY runtime-env.sh ${CONTAINER_RUNTIME_ROOT}/runtime-env.sh
COPY write-runtime-metadata.py /tmp/write-runtime-metadata.py
${MUSL_PYTHON_COPY}

RUN chmod +x ${CONTAINER_RUNTIME_ROOT}/runtime/run_openhands.py && \\
    chmod +x ${CONTAINER_RUNTIME_ROOT}/bin/python && \\
    chmod +x ${CONTAINER_RUNTIME_ROOT}/bin/pip && \\
    ln -sf python ${CONTAINER_RUNTIME_ROOT}/bin/python3 && \\
    ln -sf pip ${CONTAINER_RUNTIME_ROOT}/bin/pip3 && \\
    chmod +x ${CONTAINER_RUNTIME_ROOT}/runtime-env.sh

RUN bash -lc 'set -euo pipefail; \\
    for abi in \$SAGENT_RUNTIME_PYTHON_ABIS; do \\
      case "\$abi" in \\
        glibc) python_request="cpython-${PYTHON_VERSION}-linux-x86_64-gnu" ;; \\
        musl) \\
          if [ ! -x "\$SAGENT_RUNTIME_PYTHON_HOME_ROOT/musl/bin/python3" ]; then \\
            echo "musl Python home was not copied from the Alpine builder stage" >&2; \\
            exit 1; \\
          fi; \\
          continue ;; \\
        *) echo "Unsupported Python ABI: \$abi" >&2; exit 1 ;; \\
      esac; \\
      tmp_python_dir="\$(mktemp -d)"; \\
      uv python install --install-dir "\$tmp_python_dir" "\$python_request"; \\
      embedded_python_home="\$(find "\$tmp_python_dir" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | head -n 1)"; \\
      rm -rf "\$SAGENT_RUNTIME_PYTHON_HOME_ROOT/\$abi"; \\
      mv "\$embedded_python_home" "\$SAGENT_RUNTIME_PYTHON_HOME_ROOT/\$abi"; \\
      rm -rf "\$tmp_python_dir"; \\
      uv pip install --python "\$SAGENT_RUNTIME_PYTHON_HOME_ROOT/\$abi/bin/python3" --break-system-packages ${PACKAGE_SPECS_ESCAPED}; \\
    done'

RUN bash -lc 'set -euo pipefail; \\
    for abi in \$SAGENT_RUNTIME_PYTHON_ABIS; do \\
      if [ "\$abi" != glibc ]; then continue; fi; \\
      CUSTOM_AGENT_RUNTIME_PYTHON_ABI="\$abi"; \\
      export CUSTOM_AGENT_RUNTIME_PYTHON_ABI; \\
      . "\$SAGENT_RUNTIME_ROOT/runtime-env.sh"; \\
      "\$SAGENT_RUNTIME_ROOT/bin/python" -c "import openhands.sdk"; \\
      "\$SAGENT_RUNTIME_ROOT/bin/pip" show openhands-sdk >/dev/null; \\
      "\$SAGENT_RUNTIME_ROOT/bin/python" "\$SAGENT_RUNTIME_ROOT/runtime/run_openhands.py" --help >/dev/null; \\
    done; \\
    unset CUSTOM_AGENT_RUNTIME_PYTHON_ABI'

RUN bash -lc 'set -euo pipefail; \\
    . "\$SAGENT_RUNTIME_ROOT/runtime-env.sh"; \\
    PACKAGES_JSON='"'"'${PACKAGES_JSON}'"'"' PYTHON_ABIS="\$SAGENT_RUNTIME_PYTHON_ABIS" RUNTIME_ROOT="\$SAGENT_RUNTIME_ROOT" "\$SAGENT_RUNTIME_ROOT/bin/python" /tmp/write-runtime-metadata.py'

LABEL sagent.runtime.agent="${AGENT_NAME}" \\
      sagent.runtime.version="${RUNTIME_VERSION}" \\
      sagent.runtime.mode="self-contained"
ENV CUSTOM_AGENT_RUNTIME_ROOT=${CONTAINER_RUNTIME_ROOT}
ENV CUSTOM_AGENT_PYTHON=${CONTAINER_RUNTIME_ROOT}/bin/python
WORKDIR ${CONTAINER_RUNTIME_ROOT}
CMD ["bash"]
EOF

docker build --tag "${IMAGE_NAME}" "${BUILD_CONTEXT}"

echo "Built self-contained runtime image:"
echo "  image: ${IMAGE_NAME}"
echo "  runtime_source: ${RUNTIME_SOURCE_DIR}"
echo "  mode: self-contained"
echo "  python_abis: ${PYTHON_ABIS}"

if [[ ${PUSH_IMAGE} -eq 1 ]]; then
  docker push "${IMAGE_NAME}"
  echo "Pushed runtime image:"
  echo "  image: ${IMAGE_NAME}"
fi
