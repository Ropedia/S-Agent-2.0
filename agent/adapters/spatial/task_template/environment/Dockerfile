# ERQA task environment: CPU-only, no torch, no CUDA.
# All heavy perception (DA3 depth/3D, SAM3 detect/segment) runs on the host GPU
# server and is reached over HTTP, so this image only needs an HTTP client,
# small array/image libraries for post-processing responses, and jq for
# reading JSON replies from shell.
FROM python:3.11-slim-bookworm

# Host identity the agent runs as. The gateway runs on the HOST as this same
# uid:gid, and both it and this container read/write one shared per-trial
# /perception directory -- so the two must agree on uid or one of them cannot
# create files in the other's directories (that was Bug A: Docker auto-creates
# an absent bind-mount source as root:root, the container wrote into it as
# root, and the gateway then got EACCES on `mkdir artifacts/`, 400-ing every
# single perception call).
#
# These are ARGs with placeholder defaults, NOT hardcoded host values: the
# adapter rewrites the two default lines below with this host's real
# `id -u` / `id -g` when it generates each task directory (see
# ERQAAdapter._render_dockerfile). Keeping them as ARGs means the template
# itself stays portable -- nothing here is bound to one machine.
ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# `from perception import look` must work from ANY cwd. WORKDIR /app below only
# sets the shell's starting directory; the agent runs most of its python as
# `/usr/local/bin/python3 <<'PYEOF'` heredocs, and Python puts "" (the cwd) on
# sys.path -- not /app -- so a heredoc launched after any `cd` fails with
# ModuleNotFoundError: No module named 'perception'.
#
# Measured across two 200-question runs: this is the highest-frequency
# tool-layer error in BOTH arms at ~1.6 occurrences per trial, i.e. very likely
# the single largest source of wasted turns in the project. It looks impossible
# from the outside because `cd /app && python3 -c ...` works and `pip show`
# finds everything -- the failure only appears in the heredoc form the agent
# actually prefers (measured: 209 of 212 object-layer calls used one).
#
# Fixed here rather than by telling the agent to add sys.path.insert("/app"):
# an instruction it must remember on every call is a class of failure no
# wording removes (v3 lost 45 calls to exactly that shape).
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Pinned to keep the build reproducible and fast; this image is rebuilt once
# per task directory (400 of them), so no compilers and no large wheels.
RUN pip install --no-cache-dir \
        "numpy==2.2.6" \
        "pillow==11.1.0" \
        "requests==2.32.3"

# Idempotent: python:3.11-slim already ships some ids, and a host uid/gid may
# collide with one of them. getent is the check that works for both a name and
# a numeric id, so a pre-existing id is reused rather than failing the build.
RUN set -eux; \
    if ! getent group "$APP_GID" >/dev/null; then \
        groupadd -g "$APP_GID" app; \
    fi; \
    if ! getent passwd "$APP_UID" >/dev/null; then \
        useradd -u "$APP_UID" -g "$APP_GID" -m -s /bin/bash app; \
    fi

# Sample images (frame_0.png, frame_1.png, ...) plus perception helper.
COPY workspace/ /app/

# /app must be writable by the agent uid or `answer.txt` cannot be written and
# EVERY task scores 0 -- a far worse failure than the one this change fixes.
# tests/test.sh reads /app/answer.txt, so this is on the critical scoring path.
RUN chown -R "$APP_UID:$APP_GID" /app

# Runs as root (PID 1 is deliberately left as root -- see below) before handing
# off to the container command. Docker creates an absent bind-mount source as
# root:root *after* image build and *before* this runs, so this is the earliest
# moment at which the per-trial /perception directory exists and can be given
# to the agent uid. Doing it here rather than on the host is what solves the
# ordering problem: the host launcher never learns a trial's directory name
# (SAgent mints it at run time, long after run_job.sh has backgrounded SAgent
# and returned), but the container starting up inside that trial always has the
# directory already mounted at a fixed path.
#
# Non-fatal by construction: the disk_only arm is given no /perception mount at
# all, and must still start normally.
RUN printf '%s\n' \
        '#!/bin/sh' \
        '# Give the per-trial bind mount to the agent uid; see Dockerfile.' \
        'if [ -d /perception ]; then' \
        "    chown -h ${APP_UID}:${APP_GID} /perception 2>/dev/null || true" \
        'fi' \
        'exec "$@"' \
        > /usr/local/bin/hb-entrypoint.sh \
    && chmod 0755 /usr/local/bin/hb-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/hb-entrypoint.sh"]

# NOTE: no `USER` directive, on purpose. USER would make PID 1 -- and therefore
# the ENTRYPOINT above -- non-root, which is exactly the one thing that cannot
# chown the root-owned directory Docker just created; Bug A would simply move
# from the gateway side to the container side. The agent and verifier are run
# as APP_UID instead through SAgent's own `docker exec -u`, driven by
# `[agent] user` / `[verifier] user` in task.toml (sagent sets
# environment.default_user from those before each phase).
CMD ["sleep", "infinity"]
