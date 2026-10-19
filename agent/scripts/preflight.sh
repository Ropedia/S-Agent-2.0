#!/usr/bin/env bash
# Assert every known silent-failure mode is absent before spending hours on a run.
# Each check maps to a failure that previously produced plausible but
# meaningless results.
set -uo pipefail

SDK_PATH="${SDK_PATH:-/tmp/ohsdk/sdk}"

# Which interpreter the checks run under. Bare `python3` resolves through PATH,
# which on 2026-08-21's replacement node landed on a conda python 3.13 that
# cannot import openhands-sdk (`formatargspec` was removed from `inspect` in
# 3.11), so check 5 failed with an import error and run_job.sh -- correctly --
# refused to start. The failure looked like "vision gate closed", i.e. like the
# single most consequential silent-failure mode this script exists to catch,
# when the gate was in fact open. Pinning it to the repo's own venv makes the
# check mean what it says regardless of what PATH holds on a given machine.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PF_PY="${PF_PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PF_PY" ] || PF_PY=python3

VLLM="${VLLM_URL:-http://127.0.0.1:8077}"
PF_MODEL="${LLM_MODEL_NAME:-qwen3.6-35b-a3b}"
export VLLM PF_MODEL
PF_BACKEND_KEY="${VLLM_API_KEY:-bearer}"
PF_BACKEND_MODEL="${VLLM_MODEL:-$PF_MODEL}"
export PF_BACKEND_MODEL
PF_HEALTH_URL="${VLLM_HEALTH_URL:-$VLLM/health}"
export PF_BACKEND_KEY
PROXY_PORT="${LITELLM_PORT:-4011}"
PROXY="http://127.0.0.1:${PROXY_PORT}"
GATEWAY="${PERCEPTION_URL:-http://127.0.0.1:${PERCEPTION_PORT:-8090}}"
FAIL=0

check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  PASS  $name"
  else
    echo "  FAIL  $name"
    FAIL=1
  fi
}

echo "=== 1. services up ==="
check "LLM backend health"      curl -sf --max-time 10 -H "Authorization: Bearer $PF_BACKEND_KEY" "$PF_HEALTH_URL"
check "proxy /health/readiness" curl -sf --max-time 10 -H "Authorization: Bearer dummy-key-cf" "$PROXY/health/readiness"
check "perception gateway"      curl -sf --max-time 10 "$GATEWAY/health"

echo "=== 2. no NFS storm (vLLM ran 96x slow under one) ==="
NFS_READS=$("$PF_PY" - <<'PY'
import re, time
def snap():
    s = open('/proc/self/mountstats').read()
    # Key on the local mount POINT (/home), not the remote export path -- a
    # re-export under a different remote path (e.g. after an NFS remount)
    # must not silently read as "mount not found". A not-found mount is a
    # measurement failure, never a reading of zero.
    m = re.search(r'^device (\S+) mounted on /home with fstype nfs', s, re.MULTILINE)
    if not m:
        # Two very different situations, and reading them the same way is how a
        # real measurement failure gets waved through. If this host mounts NFS
        # *somewhere* but not at /home, the mount string moved under us and the
        # measurement has failed -- still an error. If the host mounts no NFS at
        # all (the docker host: / is ext4, nothing NFS anywhere), the storm this
        # check exists to catch is impossible, so the check is vacuous rather
        # than broken; say so explicitly instead of reading it as "0 reads".
        if re.search(r'^device \S+ mounted on \S+ with fstype nfs', s, re.MULTILINE):
            raise RuntimeError(
                "no NFS mount at /home, but this host does mount NFS elsewhere "
                "-- the /home mount string changed; fix this check before trusting it")
        return None
    dev = m.group(1)
    blk = s.split(f'device {dev} mounted on /home')[1].split('device ')[0]
    m2 = re.search(r'\sREAD:\s+(\d+)', blk)
    if not m2:
        raise RuntimeError(f"no READ counter found for /home mount (device {dev})")
    return int(m2.group(1))
try:
    a = snap()
    if a is None:
        print("NONFS")
    else:
        time.sleep(5)
        print(snap() - a)
except Exception as e:
    print(f"ERR: {e}")
PY
)
# A failed measurement (exception, empty stdout) must not be read as "0
# reads" -- require an actual non-negative integer before comparing.
if [ "$NFS_READS" = NONFS ]; then
  echo "  PASS  host mounts no NFS at all; the storm this checks for cannot happen"
elif ! [[ "$NFS_READS" =~ ^[0-9]+$ ]]; then
  echo "  FAIL  could not measure NFS reads (mount string may have changed): $NFS_READS"
  FAIL=1
elif [ "$NFS_READS" -lt 5000 ]; then
  echo "  PASS  NFS reads in 5s: $NFS_READS"
else
  echo "  FAIL  NFS reads in 5s: $NFS_READS (>=5000; look for runaway recursive greps over /home)"
  FAIL=1
fi

echo "=== 3. vLLM latency sane (was 192.6s under the NFS storm, 2.0s healthy) ==="
LAT=$("$PF_PY" - <<'PY'
import json, os, time, urllib.request
body = json.dumps({"model": os.environ.get("PF_BACKEND_MODEL", "qwen3.6-35b-a3b"),
                   "max_tokens": int(os.environ.get("PF_LATENCY_MAX_TOKENS", "8")),
                   "messages": [{"role": "user", "content": "hi"}]}).encode()
r = urllib.request.Request(os.environ.get("VLLM", "http://127.0.0.1:8077") + "/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + os.environ.get("PF_BACKEND_KEY", "bearer"),
             "User-Agent": "sagent-preflight/1.0"})
t = time.time()
try:
    urllib.request.urlopen(r, timeout=120); print(f"{time.time()-t:.1f}")
except Exception:
    print("999")
PY
)
if "$PF_PY" -c "import sys; sys.exit(0 if float('$LAT') < 30 else 1)"; then
  echo "  PASS  8-token latency ${LAT}s"
else
  echo "  FAIL  8-token latency ${LAT}s (>=30s)"
  FAIL=1
fi

echo "=== 4. native tool calling (400s without --enable-auto-tool-choice) ==="
TC=$("$PF_PY" - <<'PY'
import json, os, urllib.request
# max_tokens=1024, not a small number: this model (Qwen3.6-35B-A3B) emits a
# <think>...</think> reasoning block before the tool-call XML. A small budget
# truncates the response mid-reasoning (finish_reason=length, no tool_calls),
# which looks like missing tool-call support but is really just running out
# of room on a healthy stack -- measured 3 FAIL / 5 PASS at max_tokens=128 on
# this deployment. A genuinely missing --enable-auto-tool-choice /
# --tool-call-parser fails immediately (usually an HTTP 400) at any token
# budget, so the larger budget does not mask that failure.
body = json.dumps({"model": os.environ.get("PF_BACKEND_MODEL", "qwen3.6-35b-a3b"),
  "max_tokens": int(os.environ.get("PF_TOOLCALL_MAX_TOKENS", "1024")),
  "messages": [{"role": "user", "content": "List files in /tmp using the tool."}],
  "tools": [{"type": "function", "function": {"name": "run_bash",
     "description": "Run a bash command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"]}}}],
  "tool_choice": "auto"}).encode()
r = urllib.request.Request(os.environ.get("VLLM", "http://127.0.0.1:8077") + "/v1/chat/completions", data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer " + os.environ.get("PF_BACKEND_KEY", "bearer"),
             "User-Agent": "sagent-preflight/1.0"})
try:
    o = json.load(urllib.request.urlopen(r, timeout=300))
    fr = o["choices"][0].get("finish_reason")
    has_tc = bool(o["choices"][0]["message"].get("tool_calls"))
    print("yes" if has_tc else f"no (finish_reason={fr}, tool_calls_present={has_tc})")
except Exception as e:
    print(f"no (exception: {e})")
PY
)
[ "$TC" = yes ] && echo "  PASS  native tool_calls returned" || { echo "  FAIL  no tool_calls: $TC"; FAIL=1; }

echo "=== 5. vision gate open through proxy (the whole point) ==="
# Runs INSIDE the runtime image, not on the host. openhands-sdk 1.33.0 exists
# only there (the host venv cannot install it), and $SDK_PATH used to point at
# a host copy under /tmp -- which vanished with /tmp when the node was replaced
# on 2026-08-21, turning this check into a permanent false FAIL. The image is
# also the only place where "the SDK the configurations actually import" is a true
# statement, so testing there is more faithful, not just more robust.
# Falls back to the host interpreter + $SDK_PATH if the image is absent.
VIS_PY=$(cat <<'PY'
import os, sys
if os.environ.get("SDK_PATH"):
    sys.path.insert(0, os.environ["SDK_PATH"])
try:
    from openhands.sdk.llm.llm import LLM
    from openhands.sdk.llm import Message, TextContent, ImageContent
    llm = LLM(usage_id="pf", model="litellm_proxy/" + os.environ.get("PF_MODEL", "qwen3.6-35b-a3b"),
              base_url=os.environ["PROXY"], api_key="dummy-key-cf")
    # Send exactly ONE image (1x1 transparent PNG) and expect exactly ONE
    # image_url part in the formatted payload -- the expected count is a
    # literal so this can't silently drift if the message shape changes.
    url = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
           "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    msgs = [Message(role="user", content=[TextContent(text="x"),
                                          ImageContent(image_urls=[url])])]
    n = sum(1 for d in llm.format_messages_for_llm(msgs)
              if isinstance(d.get("content"), list)
              for p in d["content"] if p.get("type") == "image_url")
    active = llm.vision_is_active()
    if active and n == 1:
        print("yes")
    else:
        print(f"no (vision_is_active={active}, image_url_parts={n}, expected=1)")
except Exception as e:
    print(f"no (exception: {e})")
PY
)
VIS_IMAGE="${VIS_IMAGE:-sagent-oh-sdk-1.33.0:v1}"
if docker image inspect "$VIS_IMAGE" >/dev/null 2>&1; then
  # --network host so 127.0.0.1:$PROXY_PORT inside the container is the proxy
  # on this host. The check only formats a message; it never calls the model.
  # PF_MODEL must be passed explicitly: `docker run` does not inherit the host
  # env, and without it the snippet falls back to the default qwen alias --
  # which the gemini proxy does not serve, so the gate read False for every
  # non-default LLM_MODEL_NAME (2026-09-12, local). The host fallback below
  # never had this problem because PF_MODEL is exported at the top.
  VIS=$(printf '%s' "$VIS_PY" | timeout 180 docker run --rm -i --network host \
        -e "PROXY=$PROXY" -e "PF_MODEL=$PF_MODEL" \
        --entrypoint /opt/custom-agent-runtime/oh-sdk/bin/python \
        "$VIS_IMAGE" - 2>/dev/null | tail -1)
  VIS_WHERE="in image $VIS_IMAGE"
else
  VIS=$(printf '%s' "$VIS_PY" | SDK_PATH="$SDK_PATH" PROXY="$PROXY" "$PF_PY" - 2>/dev/null | tail -1)
  VIS_WHERE="on host (image $VIS_IMAGE absent)"
fi
[ "$VIS" = yes ] && echo "  PASS  vision_is_active and image_url part present ($VIS_WHERE)" || { echo "  FAIL  vision gate closed or image dropped $VIS_WHERE: $VIS"; FAIL=1; }

echo "=== 6. jobs dir off Lustre (root-squash breaks log bind mounts) ==="
mkdir -p /tmp/sagent-jobs
FSTYPE=$(df -T /tmp/sagent-jobs 2>/dev/null | tail -1 | awk '{print $2}')
# An empty/unparseable FSTYPE is not "not lustre" -- only a positively
# identified non-lustre type may pass.
if [ -z "$FSTYPE" ]; then
  echo "  FAIL  could not determine jobs dir filesystem type"
  FAIL=1
elif [ "$FSTYPE" != lustre ]; then
  echo "  PASS  jobs dir fstype=$FSTYPE"
else
  echo "  FAIL  jobs dir on lustre"
  FAIL=1
fi

echo
[ "$FAIL" -eq 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED"
exit "$FAIL"
