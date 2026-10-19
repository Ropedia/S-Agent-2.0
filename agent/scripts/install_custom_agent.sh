#!/usr/bin/env bash
# Install the custom agent into whatever sagent is on this machine.
#
# Why this exists: run_job.sh launches the agent with
#   --agent-import-path sagent.agents.custom.openhands_sdk:CustomOpenHandsSDK
# and `sagent.agents.custom` is NOT part of the public sagent package on PyPI.
# It only exists in the upstream harness project/sagent (private). Anyone without access to that
# repo gets ModuleNotFoundError at trial start.
#
# This script copies the vendored copies (vendor/sagent/, Apache-2.0, see the
# README there for provenance) into the installed sagent package, so a fresh
# clone plus `pip install sagent` is enough to run the experiments.
#
# Idempotent: re-running overwrites with the same bytes. Safe after a sagent
# upgrade, which is when you need it -- pip reinstall drops the custom module.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENDOR="$REPO_ROOT/vendor/sagent/agents_custom"

PY="${PY:-python3}"

if [ ! -f "$VENDOR/openhands_sdk.py" ]; then
  echo "ERROR: $VENDOR/openhands_sdk.py missing" >&2
  exit 1
fi

# Ask the interpreter where sagent actually lives rather than guessing: this
# box alone has three separate sagent installs.
SAGENT_PKG="$("$PY" -c 'import sagent,pathlib;print(pathlib.Path(sagent.__file__).parent)' 2>/dev/null || true)"
if [ -z "$SAGENT_PKG" ]; then
  echo "ERROR: sagent is not importable by $PY." >&2
  echo "  Install it first:  pip install sagent" >&2
  echo "  Or point PY= at the interpreter that has it." >&2
  exit 1
fi

DEST="$SAGENT_PKG/agents/custom"
mkdir -p "$DEST"
[ -f "$DEST/__init__.py" ] || : > "$DEST/__init__.py"
cp "$VENDOR/openhands_sdk.py" "$VENDOR/openhands_sdk_runner.py" "$DEST/"

echo "[install_custom_agent] sagent package: $SAGENT_PKG"
echo "[install_custom_agent] installed into: $DEST"

# Verify by importing the exact symbol run_job.sh asks for -- copying the file
# is not proof that the import path resolves.
"$PY" - <<'EOF'
import importlib, sys
try:
    m = importlib.import_module("sagent.agents.custom.openhands_sdk")
    getattr(m, "CustomOpenHandsSDK")
except Exception as exc:  # noqa: BLE001
    print(f"[install_custom_agent] FAILED to import: {exc}", file=sys.stderr)
    raise SystemExit(1)
print("[install_custom_agent] verified: "
      "sagent.agents.custom.openhands_sdk:CustomOpenHandsSDK imports")
EOF
