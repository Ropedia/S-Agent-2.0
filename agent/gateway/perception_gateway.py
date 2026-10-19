"""Thin REST gateway in front of the SpatialClaw GPU perception server.

Design goals
------------
* Scaffold agnostic: every capability is reachable with plain ``curl``.
* Paths in, paths out: images and arrays are exchanged as file paths under a
  shared work dir, because MCP / CLI scaffolds cannot pass typed tensors.
* Zero GPU work of its own: it proxies to the already running
  ``launch_gpu_server`` process over the msgpack ``/call`` protocol.
* Frame identity is enforced server side. Every array carries the absolute
  ``frame_indices`` it was produced for, persisted as a ``<array>.json``
  sidecar, and every consuming endpoint validates alignment (HTTP 409).
* Tool failure (HTTP 5xx) is distinguished from a legitimate empty result
  (HTTP 200 with ``{"found": false}``).
* Per-request isolation is opt-in: a request may set ``work_dir`` to move its
  reads/writes into a per-trial directory instead of the global work dir.
  ``work_dir`` must resolve under one of ``GATEWAY_TRIAL_ROOTS`` (see
  ``resolve_work_base``) -- otherwise a container could direct the gateway to
  write anywhere on the host.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response as StarletteResponse

# --- SpatialClaw imports (protocol only, no models) -------------------------
# Derived at runtime: this file lives in agent/gateway/; SpatialClaw is the submodule under perception/.
SPATIALCLAW_ROOT = os.environ.get(
    "SPATIALCLAW_ROOT",
    str(Path(__file__).resolve().parent.parent.parent / "perception" / "SpatialClaw"),
)
if SPATIALCLAW_ROOT not in sys.path:
    sys.path.insert(0, SPATIALCLAW_ROOT)

from spatial_agent.gpu_rpc import encode_request, decode_response  # noqa: E402

# Per-trial persistent 3D scene (named instances over one reconstruction).
# Sibling module: the gateway is launched as a script from gateway/, and the
# tests insert gateway/ into sys.path, so a plain sibling import works in
# both. scene_store never imports us back (state + math only).
import scene_store

# Needed to reproduce DA3's center-crop exactly; see the crop_box block in
# /reconstruct. Importing the upstream helpers rather than reimplementing the
# aspect table keeps the two in lockstep if DA3_TARGET_SHAPES ever changes.
from spatial_agent.gpu_models.image_resize import (  # noqa: E402
    crop_to_aspect,
    da3_pick_target_shape,
)

GATEWAY_ROOT = Path(
    os.environ.get("GATEWAY_ROOT", str(Path(__file__).resolve().parent.parent))
)
WORK_DIR = Path(os.environ.get("GATEWAY_WORK_DIR", GATEWAY_ROOT / "work"))
LOG_PATH = Path(os.environ.get("GATEWAY_LOG", GATEWAY_ROOT / "logs" / "gateway.jsonl"))
REGISTRY_PATH = Path(
    os.environ.get(
        "GPU_SERVER_REGISTRY",
        Path(SPATIALCLAW_ROOT) / "spatial_agent" / "logs" / "gpu_server.json",
    )
)

# Containers mount the shared work dir at a different path than the host
# (typically /work). Callers may use either namespace; responses echo paths
# back in whichever namespace the request used, so a container never receives
# a path it cannot open.
PATH_ALIASES = [
    alias.rstrip("/")
    for alias in os.environ.get("GATEWAY_PATH_ALIASES", "/work").split(",")
    if alias.strip()
]

# Roots a per-request ``work_dir`` is allowed to resolve under (see
# resolve_work_base). Colon-separated so a deployment can allow more than one
# tree at once -- e.g. the legacy WORK_DIR/trials tree plus wherever SAgent
# actually places a trial's <trial_dir>/perception directory (which need not
# live under WORK_DIR at all: SAgent trial dirs commonly live under something
# like /tmp/sagent-jobs/<job>/<trial>/perception). The default keeps the
# original single-root behaviour so existing callers/tests are unaffected.
TRIAL_ROOTS = [
    Path(root)
    for root in os.environ.get("GATEWAY_TRIAL_ROOTS", str(WORK_DIR / "trials")).split(
        ":"
    )
    if root.strip()
]

MAX_REQUEST_BYTES = int(os.environ.get("GATEWAY_MAX_REQUEST_BYTES", 32 * 1024 * 1024))
UPSTREAM_TIMEOUT_SEC = float(os.environ.get("GATEWAY_UPSTREAM_TIMEOUT", 600))
# 2026-09-04: 32 -> 64. MMSI-Video-Bench ships 32 frames per video and a
# two-video question is one /recon over 64 frames; at 32 that call was a
# plain 400 ("too many images"), which reads as a tool-layer error rather
# than a crash and is easy to miss in a run. The still-image benchmarks
# (ERQA, MMSI, MindCube, SPBench, SPAR) send at most a handful of frames,
# so nothing they do can reach either limit. Whether the upstream models
# (DA3, Pi3MOS) actually fit 64 frames is a GPU-memory question measured
# per deployment, not something this cap can answer.
MAX_IMAGES_PER_CALL = int(os.environ.get("GATEWAY_MAX_IMAGES", 64))

#: How many upstream GPU calls may be in flight at once.
#:
#: Every handler here is `async def`, but the work inside is blocking:
#: `call_upstream` uses `requests.post`, and the numpy paths are CPU-bound. A
#: blocking call in an async handler owns the single uvicorn event loop for its
#: whole duration, so while one reconstruct runs on the GPU the gateway cannot
#: accept, parse, or answer anything else -- not even /health. Queued callers
#: therefore do not wait, they FAIL: the agent reads a timeout as "this tool is
#: broken" and stops using it. That is the worst shape available.
#:
#: ⚠️ The defect is real and demonstrated by `tests/test_gateway_concurrency.py`
#: (removing either half of the fix reddens a test). But the number that USED to
#: sit here -- "7 of 48 tool rejections on v6 were client-side timeouts" -- was
#: false and has been removed. Classifying those observations by origin shows
#: all 156 of them (v6=66, v7=73, v8=17) are the agent running
#: `cat -n /app/perception.py`, whose source contains the word "timeout" ten
#: times. Zero are transport timeouts. They carry no error.code either, so
#: `run_metrics.call_outcomes()` never counted them among the 48 rejections --
#: the claim was structurally impossible, not merely wrong. PITFALLS 1.3,
#: fourth occurrence: a substring scan cannot tell what the agent EMITTED from
#: what it merely READ.
#:
#: The fix has two halves and needs both:
#:   1. run the blocking work in a worker thread, so the event loop stays free
#:      to accept and hold connections (see `run_in_threadpool` below);
#:   2. cap the number that actually reach the GPU with this semaphore, so the
#:      queue forms HERE -- bounded, observable, and waiting rather than
#:      failing -- instead of forming as a pile of half-open sockets.
#: Without (2), (1) alone would just move the pile-up onto the GPU server.
#:
#: Default 2: one upstream server process holds its models on one GPU and gets
#: no faster from more concurrent callers; 2 keeps it busy while one caller is
#: encoding/decoding.
#:
#: This is a PER-UPSTREAM cap, not a gateway-wide one. It used to be
#: gateway-wide, which silently capped the whole fleet: `launch_gpu_server
#: --num_gpus N` records N in the registry but shards nothing (it instantiates
#: each tool once, in one process, on torch's default device -- measured on
#: h20, 13084 MiB on cuda:0 against 1188 MiB on cuda:1), so the way to use more
#: than one GPU is to run more than one server process. With a single global
#: semaphore, six such processes would still have served two calls at a time.
#: Total capacity is now UPSTREAM_CONCURRENCY x (live servers hosting the tool).
UPSTREAM_CONCURRENCY = int(os.environ.get("GATEWAY_UPSTREAM_CONCURRENCY", "2"))

#: How long a caller may wait for a slot before giving up. Deliberately long:
#: the whole point is that a busy gateway makes callers WAIT rather than fail,
#: and a queued reconstruct behind two others can legitimately take minutes.
#: It still has to be finite -- an unbounded wait would hide a wedged upstream.
QUEUE_TIMEOUT_SEC = float(os.environ.get("GATEWAY_QUEUE_TIMEOUT", "900"))

WORK_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
for _root in TRIAL_ROOTS:
    _root.mkdir(parents=True, exist_ok=True)

TOOL_LIST = ["reconstruct", "detect", "render", "measure"]

# ---------------------------------------------------------------------------
# Errors: explicit taxonomy so callers can tell bad-input from tool-failure.
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Replace nan/inf with strings so an error body can always be encoded.

    Error details echo the offending input back, and that input may be exactly
    what could not be serialized: a caller sending ``points: [[NaN, 0.5]]``
    would otherwise get its clear 400 replaced by an unhandled encoder error
    (bare 500, no error code) -- the diagnostic destroyed by the very value it
    was describing.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class GatewayError(Exception):
    """Base for errors that map to a structured JSON response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, **detail: Any):
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_response(self) -> JSONResponse:
        payload = {"error": {"code": self.code, "message": self.message}}
        if self.detail:
            payload["error"]["detail"] = _json_safe(self.detail)
        return JSONResponse(status_code=self.status_code, content=payload)


class BadRequest(GatewayError):
    status_code = 400
    code = "bad_request"


class FrameMismatch(GatewayError):
    """Caller composed arrays that describe different frames."""

    status_code = 409
    code = "frame_mismatch"


class UpstreamUnavailable(GatewayError):
    status_code = 503
    code = "upstream_unavailable"


class ToolFailure(GatewayError):
    """The perception model itself errored. Never used for empty results."""

    status_code = 502
    code = "tool_failure"


# ---------------------------------------------------------------------------
# JSONL request log
# ---------------------------------------------------------------------------


def log_event(record: Dict[str, Any]) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    try:
        with LOG_PATH.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # logging must never break a request


def _summarize(value: Any, depth: int = 0) -> Any:
    """Compact, log-safe view of a request/response body."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {k: _summarize(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        if len(value) > 8:
            return [_summarize(v, depth + 1) for v in value[:8]] + [
                f"...(+{len(value) - 8})"
            ]
        return [_summarize(v, depth + 1) for v in value]
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


# ---------------------------------------------------------------------------
# Path handling: everything must live under WORK_DIR (no traversal escapes),
# or -- if a request opts into per-trial isolation via ``work_dir`` -- under
# one of TRIAL_ROOTS.
# ---------------------------------------------------------------------------


def _match_alias(raw: str) -> Optional[str]:
    """Return the alias prefix ``raw`` is expressed under, if any."""
    for alias in PATH_ALIASES:
        if raw == alias or raw.startswith(alias + "/"):
            return alias
    return None


def to_host_path(raw: str) -> str:
    """Rewrite a container-namespace path (/work/x) to the host work dir."""
    alias = _match_alias(raw)
    if alias is None:
        return raw
    relative = raw[len(alias) :].lstrip("/")
    return str(WORK_DIR / relative) if relative else str(WORK_DIR)


def to_caller_path(
    host_path: str, alias: Optional[str], work_base: Optional[Path] = None
) -> str:
    """Express a host path in a namespace the caller can open.

    Two independent namespacing schemes share this function:

    * The legacy ``/work`` alias (``PATH_ALIASES``): when the request carried
      an alias-prefixed path (see ``caller_alias``), echo the response back
      the same way, expressed relative to ``work_base``. When the caller sent
      no ``work_dir`` at all, ``resolve_work_base`` returns ``WORK_DIR``
      itself, so ``work_base`` here is ``WORK_DIR`` and this is byte-for-byte
      the pre-task-3 computation.

    * Per-trial ``work_dir`` with no alias in play -- the convention
      ``perception.py`` actually uses (bare names plus a ``work_dir`` field,
      never ``/work/...`` paths, so ``caller_alias`` returns ``None`` for
      it): return the artifact path relative to ``work_base`` instead of a
      raw host-absolute path, which a container cannot open (that path lives
      on the gateway host, not under the container's ``/perception`` mount).
      A relative path is exactly what ``resolve_in_work_dir`` already accepts
      back as input -- its bare-relative-path branch joins it onto
      ``work_base`` again -- so this round-trips cleanly through a later
      ``/measure`` call that passes back a ``/reconstruct`` or ``/detect``
      artifact path.

    When neither applies -- no alias, and ``work_base`` is the global
    ``WORK_DIR`` because the request set no ``work_dir`` -- this returns
    ``host_path`` unchanged: the exact pre-task-3 return value, preserved so
    the ~400 existing task definitions that never send ``work_dir`` see no
    change at all.
    """
    base = work_base if work_base is not None else WORK_DIR
    if not alias and base == WORK_DIR:
        return host_path
    try:
        relative = Path(host_path).resolve().relative_to(base.resolve())
    except ValueError:
        return host_path
    return f"{alias}/{relative}" if alias else str(relative)


#: How many neighbouring names to list back when a lookup misses. Enough to
#: show a typo ("frame0.png" vs "frame_0.png") without dumping a full artifacts
#: dir into an error body.
MISSING_PATH_LISTING_LIMIT = 12


def _missing_path_hint(base: Path, field: str) -> Dict[str, Any]:
    """Explain *why* a path is missing, not just that it is.

    A container reaching this error has almost always skipped one step: the
    frames live inside the image at ``/app``, and only a ``cp`` into
    ``/perception`` puts them where the gateway can read them. Measured over a
    100-question run, only 30 episodes ever ran that ``cp``, and the 6 episodes
    that hit "image does not exist: frame_0.png" got no indication that copying
    was the missing action -- so they retried the same call, or gave up on
    perception entirely.

    The distinguishing signal is cheap: if the work dir holds no ``frame_*.png``
    at all, no frame was ever staged, and the fix is a ``cp`` rather than a
    different filename. If some frames *are* present, the caller more likely
    mistyped one, so list what is actually there instead.
    """
    hint: Dict[str, Any] = {}
    try:
        names = sorted(p.name for p in base.iterdir() if p.is_file())
    except OSError:
        return hint

    # `frame_<N>.png` (every still-image benchmark) or `v<K>_frame_<N>.png`
    # (MMSI-Video: the video number rides in the name, the index stays
    # global). Same fullmatch rule as the container CLI's discover_frames.
    frames = [n for n in names if re.fullmatch(r"(?:v\d+_)?frame_\d+\.png", n)]
    if not frames:
        hint["staged_frames"] = []
        hint["hint"] = (
            "no frame_*.png files are staged in this work_dir, so no frame was "
            "ever copied where the gateway can read it. Run "
            "`cp /app/frame_*.png /perception/` in the container first, then "
            "retry this call."
        )
    else:
        hint["staged_frames"] = frames[:MISSING_PATH_LISTING_LIMIT]
        hint["hint"] = (
            f"{len(frames)} frame(s) are staged in this work_dir but "
            f"{field} is not one of them -- refer to a frame by its bare "
            "name, exactly as listed in staged_frames."
        )
    if len(names) > len(frames):
        hint["other_files"] = [n for n in names if n not in frames][
            :MISSING_PATH_LISTING_LIMIT
        ]
    return hint


def resolve_in_work_dir(
    raw: Any,
    *,
    must_exist: bool,
    field: str,
    work_base: Optional[Path] = None,
) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise BadRequest(f"{field} must be a non-empty path string", got=repr(raw))
    base = work_base if work_base is not None else WORK_DIR
    candidate = Path(to_host_path(raw))

    # Accept the paths we ourselves emit. Array results come back as absolute host
    # paths (".../work/artifacts/depth_ab12.npy"). Feeding one straight back used
    # to fail with "does not exist: work/work/artifacts/..." because the value was
    # joined onto work_root a second time. Two shapes need tolerating:
    #   1. the absolute path verbatim
    #   2. a relative path that already carries the work-dir name, e.g.
    #      "work/artifacts/x.npy" — what an agent produces when it copies the
    #      tail of a printed path
    # Traversal is still rejected: we only short-circuit for paths that genuinely
    # resolve inside work_root.
    if not candidate.is_absolute():
        parts = candidate.parts
        if parts and parts[0] == base.name:
            stripped = Path(*parts[1:]) if len(parts) > 1 else None
            if stripped is not None and (base / stripped).exists():
                candidate = base / stripped

    if candidate.is_absolute():
        try:
            candidate.resolve().relative_to(base.resolve())
        except ValueError:
            pass
        else:
            resolved = candidate.resolve()
            if must_exist and not resolved.is_file():
                raise BadRequest(
                    f"{field} does not exist: {raw}",
                    resolved_to=str(resolved),
                    work_dir=str(base),
                    **_missing_path_hint(base, field),
                )
            return resolved
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
        work_root = base.resolve()
    except OSError as exc:
        raise BadRequest(f"{field} could not be resolved: {exc}") from exc
    if resolved != work_root and work_root not in resolved.parents:
        raise BadRequest(
            f"{field} must be inside the shared work dir",
            work_dir=str(work_root),
            got=str(resolved),
        )
    if must_exist and not resolved.is_file():
        raise BadRequest(
            f"{field} does not exist: {raw}",
            resolved_to=str(resolved),
            work_dir=str(work_root),
            **_missing_path_hint(base, field),
        )
    return resolved


def resolve_work_base(payload: Dict[str, Any]) -> Path:
    """Resolve the per-request work base directory, defaulting to WORK_DIR.

    A request may set ``work_dir`` (a host-absolute path) to move its reads
    and writes into a per-trial directory -- e.g. SAgent's own
    ``<trial_dir>/perception`` -- instead of the global work dir. The
    container computes this path itself (it can read its own bind-mount
    source out of /proc/self/mountinfo) and sends it verbatim; the gateway
    does no alias translation for it, only containment validation.

    Security: the resolved directory (after following symlinks and collapsing
    ``..``, via Path.resolve()) must land at, or inside, one of TRIAL_ROOTS.
    Resolution happens *before* the containment check, so this defeats
    absolute-path escape, ``../`` traversal, AND symlink escape (a directory
    under an allowed root that is itself a symlink pointing elsewhere) with
    one mechanism -- the same one resolve_in_work_dir already relies on.

    Multiple roots are genuinely independent: containment is checked against
    each root's own resolved prefix in turn, using Path.parents (proper
    parent-directory containment), not a string prefix comparison. That
    matters because a string check like ``str(resolved).startswith(str(root))``
    would wrongly accept a sibling directory that merely shares a root's name
    as a prefix (e.g. root ``/x/trials`` would wrongly admit
    ``/x/trials-evil/secret``). A request that does not resolve inside root A
    is never let through merely because some other root B is also configured;
    each candidate is checked on its own merits.
    """
    raw = payload.get("work_dir")
    if raw is None:
        return WORK_DIR
    if not isinstance(raw, str) or not raw.strip():
        raise BadRequest("work_dir must be a non-empty path string", got=repr(raw))
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise BadRequest("work_dir must be an absolute path", got=raw)
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise BadRequest(f"work_dir could not be resolved: {exc}") from exc

    for root in TRIAL_ROOTS:
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved

    raise BadRequest(
        "work_dir must resolve inside one of the configured trial roots",
        got=str(resolved),
        allowed_roots=[str(r) for r in TRIAL_ROOTS],
    )


def caller_alias(payload: Dict[str, Any]) -> Optional[str]:
    """Detect the namespace a request expressed its paths in.

    Scans the string values a caller may send so responses can be echoed back
    in the same namespace the caller can actually open.
    """

    def scan(value: Any, depth: int = 0) -> Optional[str]:
        if depth > 4:
            return None
        if isinstance(value, str):
            return _match_alias(value)
        if isinstance(value, dict):
            for item in value.values():
                found = scan(item, depth + 1)
                if found:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = scan(item, depth + 1)
                if found:
                    return found
        return None

    return scan(payload)


def _validated_artifact_dir(subdir: str, base: Path) -> Path:
    """Create ``base/subdir`` and prove the *real* directory is inside ``base``.

    ``resolve_work_base`` validates the ``work_dir`` a request asked for, but
    the path actually written to is ``work_base / subdir``, and that extra
    component is attacker-controlled in exactly the same way: the container
    owns its own ``/perception`` mount, so it can replace the ``artifacts``
    (or ``renders``) *subdirectory* with a symlink pointing anywhere on the
    host. ``mkdir(parents=True, exist_ok=True)`` accepts a symlink-to-a-
    directory silently, so without this check the gateway would follow it and
    write outside the trial root as the host user -- while the response echoed
    a well-behaved in-root path (``to_caller_path`` re-derives it from the
    pre-resolution string), making the escape invisible to the caller.

    Two layers, because either one alone leaves a hole:

    * Refuse a symlink at ``base/subdir`` outright, and create the directory
      with a plain ``os.mkdir`` (never ``exist_ok`` over a symlink).
    * Re-run the *same* containment rule ``resolve_in_work_dir`` uses for
      reads -- ``Path.resolve()`` first (collapses ``..``, follows every
      symlink in the chain), then proper parent containment via
      ``Path.parents`` -- against ``base``. Checking against ``base`` rather
      than the enclosing TRIAL_ROOT is deliberate and strictly stronger: one
      trial must not write into a sibling trial either.

    ``base`` is ``WORK_DIR`` for requests that send no ``work_dir`` at all, so
    the legacy global path ``WORK_DIR/artifacts`` is validated by the same
    rule and (being a real directory inside ``WORK_DIR``) passes unchanged.
    """
    out_dir = base / subdir
    if out_dir.is_symlink():
        raise BadRequest(
            f"{subdir} must be a real directory inside the work dir, not a symlink",
            work_dir=str(base),
            subdir=subdir,
        )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:  # exists as a non-directory
        raise BadRequest(
            f"{subdir} exists but is not a directory", work_dir=str(base)
        ) from exc
    except OSError as exc:
        raise BadRequest(
            f"could not create {subdir}: {exc}", work_dir=str(base)
        ) from exc

    try:
        resolved = out_dir.resolve()
        base_resolved = base.resolve()
    except OSError as exc:
        raise BadRequest(f"{subdir} could not be resolved: {exc}") from exc
    if base_resolved not in resolved.parents:
        raise BadRequest(
            f"{subdir} must resolve inside the work dir",
            work_dir=str(base_resolved),
            got=str(resolved),
        )
    return out_dir


def new_artifact_path(
    prefix: str,
    suffix: str,
    subdir: str = "artifacts",
    work_base: Optional[Path] = None,
) -> Path:
    """Path for a fresh artifact, with its parent directory validated.

    Prefer ``artifact_writer`` for anything that actually writes: it holds the
    validated directory open by file descriptor, so the directory cannot be
    swapped for a symlink between this check and the write. This function is
    the path-only form, still safe against a symlink that is already in place.
    """
    base = work_base if work_base is not None else WORK_DIR
    out_dir = _validated_artifact_dir(subdir, base)
    return out_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"


class ArtifactWriter:
    """Writes artifact files through a held-open directory file descriptor.

    Validating a path string and then writing to that string leaves a window:
    between the check and the write, the container could replace the validated
    directory with a symlink (it owns its ``/perception`` mount), and the write
    would follow it out of the trial root. Resolving that by re-checking harder
    is not possible -- any name-based check can be invalidated after it runs.

    So the directory is opened *once*, with ``O_NOFOLLOW | O_DIRECTORY``, and
    every file is created relative to that descriptor via ``dir_fd=``. A
    descriptor refers to the inode, not the name: after it is open, renaming
    or symlinking the directory name cannot redirect the write. Individual
    files are created ``O_CREAT | O_EXCL | O_NOFOLLOW`` so a pre-planted
    symlink under our (uuid) filename cannot redirect them either.

    ``.path`` is the nominal host path, used only for responses and sidecar
    metadata -- never reopened for writing.
    """

    def __init__(self, dir_fd: int, out_dir: Path, filename: str):
        self._dir_fd = dir_fd
        self._out_dir = out_dir
        self._filename = filename

    @property
    def path(self) -> Path:
        return self._out_dir / self._filename

    def open(self, extra_suffix: str = "", mode: str = "wb"):
        """Open <filename><extra_suffix> inside the validated directory."""
        name = self._filename + extra_suffix

        def _opener(_path: str, flags: int) -> int:
            flags |= os.O_NOFOLLOW | os.O_EXCL
            return os.open(name, flags, 0o644, dir_fd=self._dir_fd)

        try:
            return open(name, mode, opener=_opener)
        except OSError as exc:
            # Reached when the directory this fd refers to was unlinked or
            # replaced after validation (the write is refused rather than
            # redirected -- see the class docstring), or when a symlink was
            # pre-planted under our filename (O_EXCL | O_NOFOLLOW).
            raise BadRequest(
                f"could not create artifact {name}: {exc}",
                work_dir=str(self._out_dir),
            ) from exc


@contextlib.contextmanager
def artifact_writer(
    prefix: str,
    suffix: str,
    subdir: str = "artifacts",
    work_base: Optional[Path] = None,
):
    """Validated, symlink-proof writer for one new artifact (plus sidecars)."""
    base = work_base if work_base is not None else WORK_DIR
    out_dir = _validated_artifact_dir(subdir, base)
    try:
        dir_fd = os.open(str(out_dir), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise BadRequest(
            f"could not open {subdir} for writing: {exc}", work_dir=str(base)
        ) from exc
    try:
        yield ArtifactWriter(
            dir_fd, out_dir, f"{prefix}_{uuid.uuid4().hex[:12]}{suffix}"
        )
    finally:
        os.close(dir_fd)


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except Exception as exc:
        raise BadRequest(f"could not read image {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Frame-identity envelope. Each saved .npy gets a <array>.json sidecar naming
# the absolute frame indices it describes, so alignment survives across calls
# and across processes.
# ---------------------------------------------------------------------------


def save_array(
    array: np.ndarray,
    prefix: str,
    frame_indices: List[int],
    kind: str,
    extra: Optional[Dict[str, Any]] = None,
    work_base: Optional[Path] = None,
) -> str:
    with artifact_writer(prefix, ".npy", work_base=work_base) as writer:
        path = writer.path
        # np.save() on a file object writes the same bytes as np.save(path);
        # allow_pickle defaults off for these plain numeric arrays either way.
        with writer.open() as handle:
            np.save(handle, array)
        sidecar = {
            "path": str(path),
            "kind": kind,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "frame_indices": [int(i) for i in frame_indices],
            "created": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            sidecar.update(extra)
        with writer.open(".json", "w") as handle:
            handle.write(json.dumps(sidecar, indent=2))
    return str(path)


def load_envelope(
    raw: Any, field: str, work_base: Optional[Path] = None
) -> Dict[str, Any]:
    """Load a saved array plus its frame-identity sidecar."""
    path = resolve_in_work_dir(raw, must_exist=True, field=field, work_base=work_base)
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.is_file():
        raise BadRequest(
            f"{field} has no frame-identity sidecar; arrays must come from "
            f"/reconstruct or /detect so their frames are known",
            expected_sidecar=str(sidecar_path),
        )
    try:
        meta = json.loads(sidecar_path.read_text())
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise BadRequest(f"could not load {field} ({path}): {exc}") from exc
    meta["array"] = array
    meta["resolved_path"] = str(path)
    return meta


def _describe(envelopes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "path": e.get("resolved_path"),
            "kind": e.get("kind"),
            "frame_indices": [int(i) for i in e.get("frame_indices", [])],
        }
        for e in envelopes
    ]


def validate_alignment(
    envelopes: List[Dict[str, Any]], frame_index: Optional[int] = None
) -> List[int]:
    """Enforce that composed arrays actually describe the same frame(s).

    This is the generic-harness equivalent of
    ``per_frame_types.PerFrameData.validate_alignment``, but it enforces the
    contract that matters rather than raw set equality: a per-frame array (a
    detection mask on frame 7) is legitimately combined with a multi-frame
    array (a reconstruction over frames [5..9]) as long as frame 7 is present
    in both. What must never happen is measuring a mask from one frame against
    geometry from a different frame -- that silently yields a plausible-looking
    but wrong number, so it is an HTTP 409.
    """
    if not envelopes:
        return []

    frame_sets = [[int(i) for i in e.get("frame_indices", [])] for e in envelopes]
    for frames, env in zip(frame_sets, envelopes):
        if not frames:
            raise FrameMismatch(
                "An input array has no frame identity recorded, so alignment "
                "cannot be verified.",
                inputs=_describe(envelopes),
            )

    shared = set(frame_sets[0])
    for frames in frame_sets[1:]:
        shared &= set(frames)
    if not shared:
        raise FrameMismatch(
            "Frame index mismatch between inputs: these arrays describe "
            "disjoint frames, so they cannot be composed. Re-run detection on "
            "a frame that the reconstruction covers.",
            inputs=_describe(envelopes),
        )

    if frame_index is not None and int(frame_index) not in shared:
        raise FrameMismatch(
            f"Frame {int(frame_index)} is not present in every input array. "
            f"Measuring a mask from one frame against geometry from another "
            f"frame would produce a wrong number.",
            requested_frame=int(frame_index),
            frames_common_to_all_inputs=sorted(shared),
            inputs=_describe(envelopes),
        )
    return sorted(shared)


# ---------------------------------------------------------------------------
# Upstream discovery + RPC. We never start or configure a model; we only read
# the registry written by the running launch_gpu_server process.
# ---------------------------------------------------------------------------


def _pid_alive(pid: Any) -> bool:
    try:
        return Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        return False


def list_upstreams(tool: str) -> List[Dict[str, Any]]:
    """Every live registry entry hosting ``tool`` (e.g. "Reconstruct")."""
    if not REGISTRY_PATH.is_file():
        raise UpstreamUnavailable(
            "GPU server registry not found", registry=str(REGISTRY_PATH)
        )
    try:
        registry = json.loads(REGISTRY_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamUnavailable(f"registry unreadable: {exc}") from exc

    found: List[Dict[str, Any]] = []
    for key, entry in (registry or {}).items():
        if not isinstance(entry, dict):
            continue
        if not _pid_alive(entry.get("pid")):
            continue  # stale entry from a dead server
        if tool and tool not in (entry.get("tools") or []):
            continue
        ip = entry.get("ip") or "127.0.0.1"
        port = entry.get("http_port")
        token = entry.get("auth_token")
        if not port or not token:
            continue
        found.append(
            {
                "id": key,
                "url": f"http://{ip}:{int(port)}",
                "auth_token": token,
                "tools": entry.get("tools") or [],
                "pid": entry.get("pid"),
                "backend": entry.get("reconstruct_backend"),
            }
        )
    return found


def find_upstream(tool: str) -> Dict[str, Any]:
    """The live upstream hosting ``tool`` with the fewest calls in flight.

    This used to return the FIRST live entry, which made a second GPU server
    process dead weight: it would be registered, healthy, and never called.

    Join-shortest-queue rather than round-robin because service times here
    differ by an order of magnitude -- measured over one SPAR run, /segment
    runs 1.70 s at the median against /render's 0.18 s. Round-robin would hand
    a server its second reconstruct while a neighbour sits idle; picking on
    depth cannot.

    The tie-break is the registry key, so with everything idle the choice is
    deterministic and therefore testable. That is not a hidden preference for
    one server: `call_upstream_async` counts a pick as in-flight the moment it
    commits, so the next caller sees depth 1 here and goes elsewhere.
    """
    candidates = list_upstreams(tool)
    if not candidates:
        raise UpstreamUnavailable(
            f"no live GPU server hosting {tool!r}", registry=str(REGISTRY_PATH)
        )
    return min(candidates, key=lambda u: (_INFLIGHT.get(u["id"], 0), u["id"]))


def call_upstream(
    deployment: str,
    method: str,
    kwargs: Dict[str, Any],
    upstream: Optional[Dict[str, Any]] = None,
) -> Any:
    """Speak the msgpack /call protocol and unwrap AgentToolOutput.

    Raises ToolFailure for model-side errors. An *empty but valid* result is
    returned normally: emptiness is a finding, not a failure.

    ``upstream`` pins which server to call. `call_upstream_async` passes the
    one it reserved a slot on; resolving again here would pick a DIFFERENT
    server, because reserving the first one raised its in-flight depth. The
    slot would then bound a server we are not talking to. Left as None (direct
    callers) it resolves its own.
    """
    tool = deployment.split("spatial_", 1)[-1]
    if upstream is None:
        upstream = find_upstream(tool)
    try:
        body = encode_request(deployment, method, kwargs)
    except Exception as exc:
        raise BadRequest(f"could not encode upstream request: {exc}") from exc

    try:
        response = requests.post(
            f"{upstream['url']}/call",
            data=body,
            headers={
                "Authorization": f"Bearer {upstream['auth_token']}",
                "Content-Type": "application/json",
            },
            timeout=UPSTREAM_TIMEOUT_SEC,
        )
    except requests.Timeout as exc:
        raise ToolFailure(
            f"upstream {deployment}.{method} timed out after "
            f"{UPSTREAM_TIMEOUT_SEC:.0f}s"
        ) from exc
    except requests.RequestException as exc:
        raise UpstreamUnavailable(f"cannot reach GPU server: {exc}") from exc

    try:
        decoded = decode_response(response.content)
    except Exception as exc:
        detail = str(exc).strip() or (
            f"{type(exc).__name__} with no message (the model raised an empty "
            f"error, commonly an invalid prompt argument)"
        )
        raise ToolFailure(
            f"upstream {deployment}.{method} failed: {detail}",
            http_status=response.status_code,
            upstream_error_type=type(exc).__name__,
        ) from exc

    if response.status_code != 200:
        raise ToolFailure(
            f"upstream returned HTTP {response.status_code} without an error body"
        )

    err = getattr(decoded, "err_msg", None)
    if err:
        raise ToolFailure(f"{deployment}.{method}: {err}")
    result = getattr(decoded, "result", decoded)
    if result is None:
        raise ToolFailure(f"{deployment}.{method} returned no result object")
    return result


#: Bounds how many blocking upstream calls run at once, PER upstream server.
#: Keyed by registry id. Created lazily on the running loop -- an anyio
#: Semaphore built at import time binds to whichever loop happens to exist
#: then, which is not the one uvicorn ends up running.
_UPSTREAM_SLOTS: Dict[str, Any] = {}

#: Calls committed to each upstream but not yet finished. This is the signal
#: `find_upstream` picks on, and it is incremented at PICK time rather than
#: after the semaphore is acquired: the pick and the increment happen with no
#: await between them, so on the single event loop they are atomic. Counting
#: only acquired calls would let a burst of callers all read depth 0, all pick
#: the same server, and queue there while its neighbours idle.
_INFLIGHT: Dict[str, int] = {}


def _upstream_slots(upstream_id: str) -> Any:
    slots = _UPSTREAM_SLOTS.get(upstream_id)
    if slots is None:
        slots = anyio.Semaphore(UPSTREAM_CONCURRENCY)
        _UPSTREAM_SLOTS[upstream_id] = slots
    return slots


async def call_upstream_async(
    deployment: str, method: str, kwargs: Dict[str, Any]
) -> Any:
    """`call_upstream`, off the event loop and behind a bounded queue.

    Two things happen here that did not before, and the pair is the point:

    * the blocking `requests.post` runs in a worker thread, so the event loop
      keeps accepting and holding connections while the GPU works. Callers
      that arrive during a long reconstruct now WAIT instead of having their
      request time out unanswered;
    * only ``UPSTREAM_CONCURRENCY`` calls reach EACH GPU server at once. The
      queue forms here, where it is bounded and shows up in the log as a wait,
      rather than as a pile of sockets the single event loop never got to.

    The server is chosen before the slot is reserved, and the choice is then
    pinned for the call. Both halves matter: choosing first is what lets the
    reservation land on the least-loaded server, and pinning is what stops
    `call_upstream` from re-resolving to a different one (our own reservation
    changed the depths it would pick on).

    A caller that waits longer than ``QUEUE_TIMEOUT_SEC`` for a slot gets
    ``UpstreamUnavailable`` -- an honest "the service is saturated", which is a
    different and more actionable message than a silent client-side timeout.
    The wait is on ONE server's queue: with several live, a caller committed to
    the shortest may still see a neighbour free up first. Waiting on all of
    them would need a multi-wait; join-shortest-queue is the standard
    approximation and costs at most one service time.
    """
    tool = deployment.split("spatial_", 1)[-1]
    upstream = find_upstream(tool)
    uid = upstream["id"]
    _INFLIGHT[uid] = _INFLIGHT.get(uid, 0) + 1
    try:
        slots = _upstream_slots(uid)
        waited_from = time.time()
        try:
            with anyio.fail_after(QUEUE_TIMEOUT_SEC):
                await slots.acquire()
        except TimeoutError as exc:
            raise UpstreamUnavailable(
                f"perception gateway saturated: waited {QUEUE_TIMEOUT_SEC:.0f}s "
                f"for a slot on {uid} to run {deployment}.{method} and none "
                f"freed up",
                upstream_concurrency=UPSTREAM_CONCURRENCY,
                upstream_id=uid,
            ) from exc

        queued_sec = time.time() - waited_from
        try:
            if queued_sec > 1.0:
                # Visible in the log so a slow run can be told apart from a
                # broken one without guessing. Silence here would make
                # saturation look like the model being slow. The id is here so
                # a lopsided fleet shows up as one server queueing.
                log_event(
                    {
                        "event": "upstream_queued",
                        "deployment": deployment,
                        "method": method,
                        "upstream_id": uid,
                        "queued_sec": round(queued_sec, 1),
                    }
                )
            return await run_in_threadpool(
                call_upstream, deployment, method, kwargs, upstream
            )
        finally:
            slots.release()
    finally:
        _INFLIGHT[uid] = _INFLIGHT.get(uid, 1) - 1


def as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CPU geometry helpers (numpy only, no GPU work).
# ---------------------------------------------------------------------------


def resize_spatial(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize of an (N, H, W[, C]) stack to (N, height, width).

    Nearest neighbour keeps 3D coordinates and confidences exact rather than
    blending across depth discontinuities.
    """
    src_h, src_w = array.shape[1], array.shape[2]
    if (src_h, src_w) == (height, width):
        return array
    rows = np.clip((np.arange(height) + 0.5) * src_h / height, 0, src_h - 1).astype(int)
    cols = np.clip((np.arange(width) + 0.5) * src_w / width, 0, src_w - 1).astype(int)
    return array[:, rows][:, :, cols]


def depth_from_points(points: np.ndarray, camera_poses: np.ndarray) -> np.ndarray:
    """Per-frame camera-space z for a world-space point map."""
    num = points.shape[0]
    depth = np.zeros(points.shape[:3], dtype=np.float32)
    for i in range(num):
        world_to_cam = np.linalg.inv(camera_poses[i].astype(np.float64))
        flat = points[i].reshape(-1, 3).astype(np.float64)
        homogeneous = np.hstack([flat, np.ones((flat.shape[0], 1))])
        depth[i] = ((world_to_cam @ homogeneous.T).T[:, 2]).reshape(points.shape[1:3])
    return depth


def finite_stats(array: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(array)
    if not finite.any():
        return {"finite_fraction": 0.0, "min": None, "max": None, "mean": None}
    values = array[finite].astype(np.float64)
    return {
        "finite_fraction": round(float(finite.mean()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
    }


def require_dict(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
    return payload


def require_list(payload: Dict[str, Any], key: str) -> List[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise BadRequest(f"{key} must be a non-empty JSON array")
    return value


def coerce_point3(raw: Any, field: str) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise BadRequest(f"{field} must be a 3-element [x, y, z] array", got=repr(raw))
    try:
        point = np.asarray([float(v) for v in raw], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{field} must contain numbers", got=repr(raw)) from exc
    if not np.isfinite(point).all():
        raise BadRequest(f"{field} must be finite", got=repr(raw))
    return point


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SpatialClaw perception gateway", version="1.0")


async def read_json_body(request: Request) -> Dict[str, Any]:
    """Read a size-limited JSON body. Bodies are small (paths, not pixels)."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > MAX_REQUEST_BYTES:
                raise BadRequest(
                    "request body too large",
                    limit_bytes=MAX_REQUEST_BYTES,
                    got_bytes=int(declared),
                )
        except ValueError:
            raise BadRequest("invalid Content-Length header")

    chunks: List[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_REQUEST_BYTES:
            raise BadRequest("request body too large", limit_bytes=MAX_REQUEST_BYTES)
        chunks.append(chunk)
    raw = b"".join(chunks)
    if not raw:
        return {}
    try:
        return require_dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise BadRequest(f"body is not valid JSON: {exc}") from exc


@app.middleware("http")
async def envelope_middleware(request: Request, call_next):
    """Log every request and convert any exception into structured JSON.

    A bare traceback never reaches the client; it goes to the JSONL log so the
    behaviour of a scaffold can still be analysed afterwards.
    """
    started = time.time()
    request_id = uuid.uuid4().hex[:12]
    try:
        response = await call_next(request)
        status = response.status_code
    except GatewayError as exc:
        status = exc.status_code
        log_event(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_sec": round(time.time() - started, 4),
                "error": {"code": exc.code, "message": exc.message, **exc.detail},
            }
        )
        return exc.to_response()
    except scene_store.SceneError as exc:
        # Scene-layer validation failures are caller errors, same as
        # BadRequest -- convert here so scene_store stays gateway-agnostic.
        as_bad = BadRequest(str(exc))
        status = as_bad.status_code
        log_event(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_sec": round(time.time() - started, 4),
                "error": {"code": as_bad.code, "message": as_bad.message},
            }
        )
        return as_bad.to_response()
    except Exception as exc:  # unexpected: log the trace, return a clean body
        status = 500
        log_event(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_sec": round(time.time() - started, 4),
                "error": {"code": "internal_error", "message": str(exc)},
                "traceback": traceback.format_exc(limit=12),
            }
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            },
        )

    log_event(
        {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_sec": round(time.time() - started, 4),
        }
    )
    response.headers["X-Request-Id"] = request_id
    return response


# --- Reply shape ----------------------------------------------------------
#
# Replies carry numeric fields, a per-target `conclusion` and `units`. The
# reply-level `note` field and its `*_note` teaching siblings are stripped
# (GATEWAY_STRIP_NOTES=1, the default): the same guidance lives in the
# profile's perception_api.md. Three event notices survive because their
# occurrence is itself a runtime signal: a failed scene registration
# (`scene_note`), a scene reset (`reset_note`) and an unchanged-position
# status (`position_note`). GATEWAY_STRIP_NOTES=0 keeps every note in the
# reply; GATEWAY_STRIP_NOTES=2 strips the event notices as well.
#
# The flag is read once at import.

STRIP_NOTES_MODE = os.environ.get("GATEWAY_STRIP_NOTES", "1")
STRIP_NOTES = STRIP_NOTES_MODE in ("1", "2")

STRIP_NOTES_KEEP = (
    frozenset()
    if STRIP_NOTES_MODE == "2"
    else frozenset({"scene_note", "reset_note", "position_note"})
)


def _is_teaching_note_key(key: Any) -> bool:
    return (
        isinstance(key, str)
        and (key == "note" or key.endswith("_note"))
        and key not in STRIP_NOTES_KEEP
    )


def _strip_note_fields(value: Any) -> Any:
    """Recursively drop `note` and teaching `*_note` keys (see above)."""
    if isinstance(value, dict):
        return {
            k: _strip_note_fields(v)
            for k, v in value.items()
            if not _is_teaching_note_key(k)
        }
    if isinstance(value, list):
        return [_strip_note_fields(v) for v in value]
    return value


if STRIP_NOTES:

    @app.middleware("http")
    async def strip_notes_middleware(request: Request, call_next):
        response = await call_next(request)
        if response.status_code != 200:
            return response
        if not response.headers.get("content-type", "").startswith("application/json"):
            return response
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunks)
        try:
            content = json.loads(body)
        except ValueError:
            # Not actually JSON despite the header. The original response's
            # body iterator is consumed, so rebuild it byte-identical.
            headers = dict(response.headers)
            headers.pop("content-length", None)
            return StarletteResponse(content=body, status_code=200, headers=headers)
        stripped = JSONResponse(status_code=200, content=_strip_note_fields(content))
        for name, val in response.headers.items():
            if name.lower() not in ("content-length", "content-type"):
                stripped.headers[name] = val
        # Auditable marker: a run can be verified against a gateway restarted
        # without the flag by curling any endpoint and checking this header.
        stripped.headers["x-notes-stripped"] = STRIP_NOTES_MODE
        return stripped

    log_event({"event": "strip_notes_enabled"})


def _round3(vec: Any) -> List[float]:
    return [round(float(v), 3) + 0.0 for v in np.asarray(vec, dtype=np.float64)]


def _ground_axes(axes: Dict[str, Any], down: np.ndarray) -> Dict[str, np.ndarray]:
    """An observer's gravity-aligned ground frame: forward projected onto the
    ground (image-up fallback for a near-vertical view), up = world up."""
    fwd_g = axes["forward_w"] - down * float(axes["forward_w"] @ down)
    if float(np.linalg.norm(fwd_g)) < 0.2:
        fwd_g = axes["up_w"] - down * float(axes["up_w"] @ down)
    fwd_g = fwd_g / np.linalg.norm(fwd_g)
    right_g = np.cross(down, fwd_g)
    return {
        "forward_w": fwd_g,
        "right_w": right_g / np.linalg.norm(right_g),
        "up_w": -down,
    }


@app.get("/health")
async def health() -> JSONResponse:
    try:
        upstream = find_upstream("Reconstruct")
    except UpstreamUnavailable as exc:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": {"code": exc.code, "message": exc.message, **exc.detail},
            },
        )
    reachable = False
    upstream_health: Any = None
    try:
        probe = requests.get(f"{upstream['url']}/health", timeout=10)
        reachable = probe.status_code == 200
        upstream_health = probe.json() if reachable else probe.status_code
    except requests.RequestException as exc:
        upstream_health = str(exc)

    # How many servers the fan-out actually has, not just that ONE answered.
    #
    # Without this, a fleet that was meant to be six shards and came up as one
    # reports a cheerful ok=true -- the gateway only ever probes the server it
    # would pick right now. That is the same shape of silent single-card
    # degradation as `launch_gpu_server --num_gpus 2` claiming two GPUs while
    # loading every model onto cuda:0, and it is invisible for exactly the same
    # reason: nothing ever printed the count. `upstream_count` is what an
    # operator greps after starting the shards.
    fleet = list_upstreams("Reconstruct")
    return JSONResponse(
        status_code=200 if reachable else 503,
        content={
            "ok": reachable,
            "upstream": upstream["url"],
            "upstream_pid": upstream["pid"],
            "upstream_tools": upstream["tools"],
            "upstream_backend": upstream["backend"],
            "upstream_health": upstream_health,
            "upstream_count": len(fleet),
            "upstream_ids": [u["id"] for u in fleet],
            "upstream_concurrency": UPSTREAM_CONCURRENCY,
            "capacity": UPSTREAM_CONCURRENCY * len(fleet),
            "tools": TOOL_LIST,
            "work_dir": str(WORK_DIR),
        },
    )


@app.post("/reconstruct")
async def reconstruct(request: Request) -> JSONResponse:
    """Metric 3D reconstruction over a list of image paths.

    Body: {"images": ["/work/f0.png", ...], "frame_indices": [0, 1]}
    Returns metric scale, per-frame extrinsics + intrinsics inline, and paths
    to .npy point maps / depth / confidence, each with a frame sidecar.
    """
    payload = await read_json_body(request)
    body, _extras = await _reconstruct_core(payload)
    return JSONResponse(status_code=200, content=body)


async def _reconstruct_core(payload: Dict[str, Any]):
    """Shared implementation of /reconstruct and /recon.

    Returns (reply_body, extras): extras carries what /recon needs to
    register the persistent scene (host-side artifact paths, poses,
    resolved image paths); /reconstruct ignores it, and its reply is
    byte-identical to what the pre-refactor handler produced. Kept below
    the /reconstruct handler on purpose: test_gateway_paths pins the
    camera_poses save_array call inside the reconstruct..detect span.
    """
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    raw_images = require_list(payload, "images")
    if len(raw_images) > MAX_IMAGES_PER_CALL:
        raise BadRequest(
            "too many images in one call",
            limit=MAX_IMAGES_PER_CALL,
            got=len(raw_images),
        )
    paths = [
        resolve_in_work_dir(
            p, must_exist=True, field=f"images[{i}]", work_base=work_base
        )
        for i, p in enumerate(raw_images)
    ]

    frame_indices = payload.get("frame_indices")
    if frame_indices is None:
        frame_indices = list(range(len(paths)))
    if not isinstance(frame_indices, list) or len(frame_indices) != len(paths):
        raise BadRequest(
            "frame_indices must be a list the same length as images",
            images=len(paths),
            frame_indices=frame_indices,
        )
    try:
        frame_indices = [int(i) for i in frame_indices]
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame_indices must be integers") from exc
    if len(set(frame_indices)) != len(frame_indices):
        raise BadRequest("frame_indices must be unique", frame_indices=frame_indices)

    images = [load_image(p) for p in paths]
    result = await call_upstream_async(
        "spatial_Reconstruct",
        "reconstruct",
        {"video_frames": images, "frame_indices": frame_indices},
    )

    points = as_numpy(getattr(result, "points", None))
    poses = as_numpy(getattr(result, "camera_poses", None))
    confidence = as_numpy(getattr(result, "confidence", None))
    intrinsics = as_numpy(getattr(result, "_intrinsics", None))
    if points is None or poses is None:
        raise ToolFailure("reconstruction returned no point map or camera poses")

    # Model runs at its own resolution; resample to the input frames so point
    # maps index the same pixels as detection masks.
    #
    # CRITICAL: DA3 does not merely resize. `_preprocess_frames_da3` **center-crops**
    # every frame to the nearest of 8 fixed aspects (DA3_TARGET_SHAPES) before
    # resizing, and snaps the whole batch to the *first* frame's target. Resampling
    # the returned point map back over the uncropped dimensions therefore stretches
    # a cropped field of view across the full frame, offsetting every 3D coordinate.
    # Measured on this task set: 48 of 133 ERQA frames (36%) hit a cropped aspect,
    # with vertical stretch up to 1.197x — so `/measure` returned plausible wrong
    # numbers on a third of the data. We reproduce DA3's crop here and report it, so
    # point maps and masks share one pixel grid.
    crop_box = None
    try:
        _tw, _th = da3_pick_target_shape(images[0].width, images[0].height)
        _target_aspect = _tw / _th
        if abs(images[0].width / images[0].height - _target_aspect) > 1e-3:
            _cropped = crop_to_aspect(images[0], _target_aspect)
            crop_box = {
                "left": (images[0].width - _cropped.width) // 2,
                "top": (images[0].height - _cropped.height) // 2,
                "width": _cropped.width,
                "height": _cropped.height,
                "note": (
                    "DA3 center-cropped to this box before inference. Point maps, "
                    "depth and intrinsics are on the CROPPED grid: add left/top to "
                    "convert a crop pixel to an original-image pixel."
                ),
            }
            target_h, target_w = _cropped.height, _cropped.width
        else:
            target_h, target_w = images[0].height, images[0].width
    except Exception:  # noqa: BLE001 - never fail a call over crop bookkeeping
        target_h, target_w = images[0].height, images[0].width
    model_h, model_w = int(points.shape[1]), int(points.shape[2])
    points = resize_spatial(points.astype(np.float32), target_h, target_w)
    if confidence is not None:
        confidence = resize_spatial(confidence.astype(np.float32), target_h, target_w)

    # Intrinsics were estimated at model resolution; rescale to input pixels.
    scaled_intrinsics = None
    if intrinsics is not None and intrinsics.ndim == 3:
        scaled_intrinsics = intrinsics.astype(np.float64).copy()
        scaled_intrinsics[:, 0, :] *= target_w / model_w
        scaled_intrinsics[:, 1, :] *= target_h / model_h

    depth = depth_from_points(points, poses.astype(np.float64))

    # crop_box travels in the sidecar too: /bev composes these arrays with
    # masks detected on the ORIGINAL grid, in a later call that never saw this
    # reply, so the crop must be recoverable from the artifact alone.
    _crop_extra = {"crop_box": crop_box} if crop_box else {}
    points_path = save_array(
        points,
        "points",
        frame_indices,
        "points",
        extra={"layout": "(N, H, W, 3) world-space XYZ in metres", **_crop_extra},
        work_base=work_base,
    )
    depth_path = save_array(
        depth,
        "depth",
        frame_indices,
        "depth",
        extra={"layout": "(N, H, W) camera-space z in metres", **_crop_extra},
        work_base=work_base,
    )
    paths_out = {"points": points_path, "depth": depth_path}
    if confidence is not None:
        paths_out["confidence"] = save_array(
            confidence,
            "confidence",
            frame_indices,
            "confidence",
            extra={"layout": "(N, H, W) reconstruction confidence"},
            work_base=work_base,
        )
    # Camera poses on disk as well as inline. The inline "camera_poses" field
    # below is unchanged (same key, same 4x4 c2w values, same rounding) -- this
    # is an additional "arrays" entry, so no existing caller sees a difference.
    # It exists because geom.py's `project --extrinsic` takes a .npy path, and
    # extrinsics were the one input /reconstruct returned only as JSON: an
    # agent following the documented example had to serialize the matrix
    # itself, which the instruction never told it to do. Saved as the full
    # (N, 4, 4) stack (not per-frame files) to match how points/depth are
    # stored, and indexed the same way -- position i is frame_indices[i].
    paths_out["camera_poses"] = save_array(
        poses.astype(np.float64),
        "camera_poses",
        frame_indices,
        "camera_poses",
        extra={
            "layout": "(N, 4, 4) camera-to-world SE(3), metres; row i is frame_indices[i]",
        },
        work_base=work_base,
    )
    host_arrays = dict(paths_out)
    paths_out = {
        k: to_caller_path(v, alias, work_base=work_base) for k, v in paths_out.items()
    }

    body = {
        "found": True,
        "frame_indices": frame_indices,
        "num_frames": len(frame_indices),
        "metric_scale": float(getattr(result, "metric_scale", 1.0)),
        "image_size": {"height": target_h, "width": target_w},
        # Non-null when DA3's aspect crop shrank the usable frame. `image_size` is
        # then the CROPPED grid, not the file's dimensions, so a mask detected on the
        # full image must be offset by (left, top) before it indexes this point map.
        "crop_box": crop_box,
        "model_size": {"height": model_h, "width": model_w},
        "arrays": paths_out,
        "camera_poses": {
            str(fi): np.asarray(poses[i], dtype=float).round(6).tolist()
            for i, fi in enumerate(frame_indices)
        },
        "camera_poses_note": "4x4 camera-to-world, metres; keyed by absolute frame index",
        "intrinsics": (
            {
                str(fi): scaled_intrinsics[i].round(4).tolist()
                for i, fi in enumerate(frame_indices)
            }
            if scaled_intrinsics is not None
            else None
        ),
        "intrinsics_note": "3x3 pinhole K rescaled to the input image size",
        "poses_finite": bool(np.isfinite(poses).all()),
        "depth_stats": finite_stats(depth),
    }
    extras = {
        "work_base": work_base,
        "alias": alias,
        "image_paths": [str(p) for p in paths],
        "frame_indices": frame_indices,
        "poses": poses.astype(np.float64),
        "crop_box": crop_box,
        "host_arrays": host_arrays,
        "image_size": {"height": target_h, "width": target_w},
        "metric_scale": body["metric_scale"],
        # In-memory references for /recon's scene-side extras (covisibility
        # check); /reconstruct ignores them. intrinsics may be None.
        "points": points,
        "depth": depth,
        "intrinsics": scaled_intrinsics,
    }
    return body, extras


@app.post("/detect")
async def detect(request: Request) -> JSONResponse:
    """Open-vocabulary detection + segmentation on one image.

    Body: {"image": "/work/f0.png", "prompt": "red car", "frame_index": 0}
    A prompt that matches nothing is HTTP 200 with {"found": false, "n": 0} --
    an honest negative observation, not an error. Only a model/transport
    failure is 5xx.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    image_path = resolve_in_work_dir(
        payload.get("image"), must_exist=True, field="image", work_base=work_base
    )

    prompt = payload.get("prompt")
    points = payload.get("points")
    boxes = payload.get("bounding_boxes")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise BadRequest("prompt must be a non-empty string when provided")
    if prompt is None and points is None and boxes is None:
        raise BadRequest("provide at least one of: prompt, points, bounding_boxes")

    try:
        frame_index = int(payload.get("frame_index", 0))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame_index must be an integer") from exc

    image = load_image(image_path)
    kwargs: Dict[str, Any] = {"image_source": image}
    if prompt is not None:
        kwargs["prompt"] = prompt
    if points is not None:
        if not isinstance(points, list) or not all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in points
        ):
            raise BadRequest("points must be [[x, y], ...] in normalized 0-1 coords")
        try:
            normalized = [[float(x), float(y)] for x, y in points]
        except (TypeError, ValueError) as exc:
            raise BadRequest(
                "points must contain numbers", got=repr(points)[:200]
            ) from exc
        # np.isfinite rejects nan/inf, which would otherwise pass the range
        # test below (nan compares False) only to break JSON serialization.
        if any(
            not np.isfinite(v) or not (0.0 <= v <= 1.0)
            for pair in normalized
            for v in pair
        ):
            raise BadRequest(
                "points must be normalized to 0-1 (divide pixel x by image "
                "width and pixel y by image height)",
                got=normalized[:4],
            )
        kwargs["points"] = normalized
        kwargs["point_labels"] = payload.get("point_labels") or [1] * len(points)
        kwargs["obj_id"] = int(payload.get("obj_id", 0))
    if boxes is not None:
        if not isinstance(boxes, list) or not all(
            isinstance(b, (list, tuple)) and len(b) == 4 for b in boxes
        ):
            raise BadRequest(
                "bounding_boxes must be [[xmin, ymin, w, h], ...] normalized 0-1"
            )
        try:
            normalized_boxes = [[float(v) for v in b] for b in boxes]
        except (TypeError, ValueError) as exc:
            raise BadRequest(
                "bounding_boxes must contain numbers", got=repr(boxes)[:200]
            ) from exc
        # The 0-1 constraint the error message above has always claimed, now
        # actually enforced -- symmetrically with the points branch. Without
        # it, a pixel-space box (the natural mistake, since /detect *returns*
        # pixel box_xyxy) reached SAM3 and came back as an HTTP 500
        # "AssertionError ... commonly an invalid prompt argument", pointing
        # the caller at the prompt when the real fix is to normalize.
        # Note xywh, not xyxy: w/h are extents, so xmin+w (not w alone) is
        # what must stay within the frame.
        for box in normalized_boxes:
            xmin, ymin, width, height = box
            bad = (
                not all(np.isfinite(v) for v in box)
                or not (0.0 <= xmin <= 1.0)
                or not (0.0 <= ymin <= 1.0)
                or not (0.0 < width <= 1.0)
                or not (0.0 < height <= 1.0)
                or xmin + width > 1.0
                or ymin + height > 1.0
            )
            if bad:
                raise BadRequest(
                    "bounding_boxes must be [[xmin, ymin, w, h], ...] normalized "
                    "to 0-1 (divide xmin and w by image width, ymin and h by "
                    "image height; w and h are extents, not corners, and "
                    "xmin+w / ymin+h must stay within 1.0)",
                    got=normalized_boxes[:4],
                )
        kwargs["bounding_boxes"] = normalized_boxes
        kwargs["bounding_box_labels"] = payload.get("bounding_box_labels") or [1] * len(
            boxes
        )

    result = await call_upstream_async("spatial_SAM3", "detect", kwargs)

    masks = as_numpy(getattr(result, "masks", None))
    boxes_out = as_numpy(getattr(result, "boxes", None))
    scores = as_numpy(getattr(result, "scores", None))
    labels = getattr(result, "labels", None) or []

    num = 0 if masks is None or masks.ndim != 3 else int(masks.shape[0])
    if num == 0:
        # Legitimate empty observation. Deliberately HTTP 200.
        return JSONResponse(
            status_code=200,
            content={
                "found": False,
                "n": 0,
                "frame_indices": [frame_index],
                "image": to_caller_path(str(image_path), alias, work_base=work_base),
                "prompt": prompt,
                "boxes": [],
                "scores": [],
                "labels": [],
                "arrays": {},
                "message": (
                    "The detector ran successfully and found no match for this "
                    "prompt in this image. This is an observation, not a failure: "
                    "try a different wording, a broader category, or point prompts."
                ),
            },
        )

    masks_bool = masks.astype(bool)
    mask_path = save_array(
        masks_bool,
        "masks",
        [frame_index],
        "masks",
        extra={
            "layout": "(n_instances, H, W) boolean instance masks",
            "prompt": prompt,
            "image": str(image_path),
        },
        work_base=work_base,
    )

    height, width = int(masks_bool.shape[1]), int(masks_bool.shape[2])
    pixels = float(height * width)
    per_instance = []
    for i in range(num):
        area = int(masks_bool[i].sum())
        # Boxes are derived from the mask itself, not taken from upstream.
        # The upstream SAM3 wrapper converts the model's top-left `xywh`
        # (sam3_model.py:288-293) as though it were `cxcywh`, so its box has the
        # right size but is offset by (-w/2, -h/2). The mask is authoritative.
        rows = np.nonzero(masks_bool[i].any(axis=1))[0]
        cols = np.nonzero(masks_bool[i].any(axis=0))[0]
        box = (
            [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])]
            if rows.size and cols.size
            else None
        )
        per_instance.append(
            {
                "index": i,
                "label": labels[i] if i < len(labels) else prompt,
                "score": (
                    round(float(scores[i]), 6)
                    if scores is not None and i < scores.shape[0]
                    else None
                ),
                "box_xyxy": box,
                "box_xyxy_upstream": (
                    np.asarray(boxes_out[i], dtype=float).round(2).tolist()
                    if boxes_out is not None and i < boxes_out.shape[0]
                    else None
                ),
                "mask_pixels": area,
                "mask_coverage": round(area / pixels, 6) if pixels else 0.0,
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "found": True,
            "n": num,
            "frame_indices": [frame_index],
            "image": to_caller_path(str(image_path), alias, work_base=work_base),
            "prompt": prompt,
            "mask_size": {"height": height, "width": width},
            "arrays": {"masks": to_caller_path(mask_path, alias, work_base=work_base)},
            "box_format": "xyxy pixels, derived from the mask extent",
            "instances": per_instance,
            "boxes": [d["box_xyxy"] for d in per_instance],
            "scores": [d["score"] for d in per_instance],
            "labels": [d["label"] for d in per_instance],
        },
    )


_SEGMENT_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)


def _segment_overlay(
    base_img: Image.Image, masks: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Color-fill every instance mask and stamp its 0-based id at the centroid.

    Ids are stamped after all fills so no label is buried under a later mask;
    fill order follows the array order (largest area first upstream), so small
    instances stay visible on top of large ones.
    """
    import colorsys

    from PIL import ImageDraw, ImageFont

    base = np.asarray(base_img, dtype=np.float32)
    n = int(masks.shape[0])
    colors = [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 1.0))
        for i in range(n)
    ]
    blended = base.copy()
    for i in range(n):
        mask = masks[i].astype(bool)
        tint = np.array(colors[i], dtype=np.float32)
        blended[mask] = (1.0 - alpha) * blended[mask] + alpha * tint
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    font = None
    for candidate in _SEGMENT_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(candidate, max(14, base.shape[0] // 40))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    for i in range(n):
        ys, xs = np.nonzero(masks[i])
        if not ys.size:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(i)
        bbox = draw.textbbox((cx, cy), label, font=font)
        draw.rectangle(
            [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill=colors[i]
        )
        draw.text(
            (cx, cy),
            label,
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="black",
        )
    return out


@app.post("/segment")
async def segment(request: Request) -> JSONResponse:
    """Prompted segmentation of NAMED objects, with an ID-stamped overlay.

        {"image": "/work/f0.png", "frame_index": 0,
         "prompts": ["statue", "red mat", "wooden stick"]}

    Runs one open-vocabulary detection per prompt (all instances of each
    concept), merges the masks across prompts, and IoU-dedupes them, so
    two prompts naming the same object yield one instance. "prompts" is
    REQUIRED: name the objects the question mentions — in the stem AND
    the answer options. (The former class-agnostic automatic mode is
    gone: blanket segment-everything filled downstream maps with
    anonymous "region" labels; naming the objects gives every mask a
    real label and keeps the map to what the question is about.)

    The reply carries an ID-stamped color overlay PNG; the stamped ids
    are the 0-based instance indices of the masks array, so they feed
    /measure directly, and each instance names the prompt that produced
    it in "label".
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    image_path = resolve_in_work_dir(
        payload.get("image"), must_exist=True, field="image", work_base=work_base
    )
    try:
        frame_index = int(payload.get("frame_index", 0))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame_index must be an integer") from exc

    kwargs: Dict[str, Any] = {}
    if payload.get("max_masks") is not None:
        try:
            as_float = float(payload["max_masks"])
        except (TypeError, ValueError) as exc:
            raise BadRequest("max_masks must be an integer") from exc
        if not as_float.is_integer():
            raise BadRequest("max_masks must be an integer", got=payload["max_masks"])
        value = int(as_float)
        if not (1 <= value <= 100):
            raise BadRequest("max_masks must be in 1..100", got=value)
        kwargs["max_masks"] = value
    if payload.get("nms_iou") is not None:
        try:
            value = float(payload["nms_iou"])
        except (TypeError, ValueError) as exc:
            raise BadRequest("nms_iou must be a number") from exc
        if not (0.0 <= value <= 1.0):
            raise BadRequest("nms_iou must be in 0..1", got=value)
        kwargs["nms_iou"] = value

    raw_prompts = payload.get("prompts")
    if raw_prompts is None:
        raise BadRequest(
            "prompts is required: list the objects to segment by name — every "
            "object the question mentions, in the stem and the answer options "
            '(e.g. {"prompts": ["bed", "wooden shelf"]})'
        )
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise BadRequest("prompts must be a non-empty list of strings")
    if len(raw_prompts) > 24:
        raise BadRequest("at most 24 prompts per call", got=len(raw_prompts))
    prompt_list: List[str] = []
    seen_prompts = set()
    for p in raw_prompts:
        if not isinstance(p, str) or not p.strip():
            raise BadRequest("every prompt must be a non-empty string", got=repr(p))
        if len(p) > 80:
            raise BadRequest("prompt too long (max 80 chars)", got=p[:80])
        cleaned = p.strip()
        if cleaned.lower() not in seen_prompts:
            seen_prompts.add(cleaned.lower())
            prompt_list.append(cleaned)

    image = load_image(image_path)
    # One open-vocabulary detection per prompt, merged. A concept that
    # matches nothing contributes nothing (that is an observation, not a
    # failure); two prompts finding the same object collapse to one
    # instance via mask-IoU NMS, keeping the higher-scoring label.
    nms_iou = float(kwargs.get("nms_iou", 0.7))
    candidates = []
    for prompt in prompt_list:
        # The DETECTOR always gets natural language: snake_case degrades
        # open-vocab detection badly (MMSI_771 smoke14a: "light_switch"
        # hallucinated 2 boxes on blank wall and split one plate into 3
        # slivers; "light switch" found nothing / one clean instance).
        # The caller's original string stays as the label and the scene
        # instance name — only the detection text is normalized.
        det_prompt = " ".join(re.sub(r"[_\-]+", " ", prompt).split()) or prompt
        result = await call_upstream_async(
            "spatial_SAM3", "detect", {"image_source": image, "prompt": det_prompt}
        )
        pm = as_numpy(getattr(result, "masks", None))
        ps = as_numpy(getattr(result, "scores", None))
        if pm is None or pm.ndim != 3:
            continue
        for i in range(pm.shape[0]):
            score = float(ps[i]) if ps is not None and i < ps.shape[0] else 0.0
            candidates.append((pm[i].astype(bool), score, prompt))
    candidates.sort(key=lambda c: -c[1])
    kept = []
    for mask, score, label in candidates:
        duplicate = False
        for kmask, _, _ in kept:
            inter = np.logical_and(mask, kmask).sum()
            if inter and inter / np.logical_or(mask, kmask).sum() >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append((mask, score, label))
    kept = kept[: kwargs.get("max_masks", 30)]
    kept.sort(key=lambda c: -int(c[0].sum()))
    if kept:
        masks = np.stack([m for m, _, _ in kept])
        scores = np.asarray([s for _, s, _ in kept], dtype=np.float32)
        labels = [lbl for _, _, lbl in kept]
    else:
        masks, scores, labels = None, None, []

    num = 0 if masks is None or masks.ndim != 3 else int(masks.shape[0])
    if num == 0:
        return JSONResponse(
            status_code=200,
            content={
                "found": False,
                "n": 0,
                "frame_indices": [frame_index],
                "image": to_caller_path(str(image_path), alias, work_base=work_base),
                "instances": [],
                "arrays": {},
                "prompts": prompt_list,
                "message": (
                    "None of the prompts matched anything in this image. This "
                    "is an observation, not a failure: try different wording "
                    "or broader categories."
                ),
            },
        )

    masks_bool = masks.astype(bool)
    mask_path = save_array(
        masks_bool,
        "segmasks",
        [frame_index],
        "masks",
        extra={
            "layout": "(n_instances, H, W) boolean instance masks",
            "source": "segment_prompted",
            "prompts": prompt_list,
            # Per-instance labels, aligned with mask order. /bev reads these to
            # name objects on the map; before this they existed only in the
            # HTTP reply, which later calls cannot see.
            "labels": [
                str(labels[i]) if i < len(labels) else "region" for i in range(num)
            ],
            "image": str(image_path),
        },
        work_base=work_base,
    )
    overlay_img = _segment_overlay(image, masks_bool)
    with artifact_writer(
        "segment_overlay", ".png", subdir="renders", work_base=work_base
    ) as writer:
        overlay_path = writer.path
        with writer.open() as handle:
            overlay_img.save(handle, format="PNG")

    height, width = int(masks_bool.shape[1]), int(masks_bool.shape[2])
    pixels = float(height * width)
    per_instance = []
    for i in range(num):
        mask = masks_bool[i]
        area = int(mask.sum())
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        box = (
            [float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1])]
            if rows.size and cols.size
            else None
        )
        ys, xs = np.nonzero(mask)
        per_instance.append(
            {
                "id": i,
                "label": labels[i] if i < len(labels) else "region",
                "box_xyxy": box,
                "centroid_xy": [int(xs.mean()), int(ys.mean())] if ys.size else None,
                "mask_pixels": area,
                "mask_coverage": round(area / pixels, 6) if pixels else 0.0,
                "score": (
                    round(float(scores[i]), 6)
                    if scores is not None and i < scores.shape[0]
                    else None
                ),
            }
        )

    # Scene-conditional (new configurations only — no scene exists unless /recon ran):
    # each labeled object becomes a named scene instance, derived from the
    # newest masks + the scene's reconstruction.
    scene_extra: Dict[str, Any] = {}
    scene = _scene_for(work_base)
    if scene is not None:
        scene_extra = _segment_scene_extra(scene, work_base, prompt_list)
        # Per-instance motion tag from the Pi3MOS dynamic mask saved at
        # /recon: an instance is "moving" when the dynamic mask covers a
        # majority of its pixels in this frame. Absent when the scene has
        # no dynamic artifact (older scene / side-channel down).
        motion_of: Dict[int, float] = {}
        dyn_env = _optional_artifact(
            "dynamic", [frame_index], "dynamic", work_base, contains=True
        )
        if dyn_env is not None:
            dyn = np.asarray(dyn_env["array"]).astype(bool)
            dyn_frames = [int(i) for i in dyn_env.get("frame_indices") or []]
            if dyn.ndim == 3 and frame_index in dyn_frames:
                dyn_frame = dyn[dyn_frames.index(frame_index)]
                dcrop = dyn_env.get("crop_box") or {}
                d_left, d_top = int(dcrop.get("left", 0)), int(dcrop.get("top", 0))
                dg_h, dg_w = int(dyn_frame.shape[0]), int(dyn_frame.shape[1])
                for i in range(num):
                    ys, xs = np.nonzero(masks_bool[i])
                    xc, yc = xs - d_left, ys - d_top
                    ok_px = (xc >= 0) & (xc < dg_w) & (yc >= 0) & (yc < dg_h)
                    if not ok_px.any():
                        continue
                    motion_of[i] = float(dyn_frame[yc[ok_px], xc[ok_px]].mean())
        # Scene configurations verify masks on the OVERLAY and read every spatial
        # fact through instance names, so the pixel-space numbers (boxes,
        # areas, scores) are withheld — comparing box heights or mask
        # areas is eyeballing in disguise. centroid_xy stays: it is the
        # legitimate pixel source for add_instance --pixel.
        return JSONResponse(
            status_code=200,
            content={
                "found": True,
                "n": num,
                "frame_indices": [frame_index],
                "overlay": to_caller_path(
                    str(overlay_path), alias, work_base=work_base
                ),
                "prompts": prompt_list,
                "id_note": (
                    "The number stamped on each colored region of the "
                    "overlay is that instance's id — drop takes "
                    "f<N>:<id>; centroid_xy is that region's pixel "
                    "centre (usable for add_instance --pixel)."
                ),
                **(
                    {
                        "motion_note": (
                            "motion says whether the OBJECT ITSELF moves "
                            "between frames (camera motion never sets it). "
                            "A moving object's cross-frame fused position "
                            "is unreliable: read it per frame "
                            "(f<N>:<label>) and use /motion for how it "
                            "moved."
                        )
                    }
                    if motion_of
                    else {}
                ),
                "instances": [
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "centroid_xy": row["centroid_xy"],
                        **(
                            (
                                {
                                    "motion": (
                                        "moving"
                                        if motion_of[row["id"]]
                                        >= _DYNAMIC_INSTANCE_FRAC
                                        else "static"
                                    )
                                }
                            )
                            if row["id"] in motion_of
                            else {}
                        ),
                    }
                    for row in per_instance
                ],
                **scene_extra,
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "found": True,
            "n": num,
            "frame_indices": [frame_index],
            "image": to_caller_path(str(image_path), alias, work_base=work_base),
            "mask_size": {"height": height, "width": width},
            "arrays": {"masks": to_caller_path(mask_path, alias, work_base=work_base)},
            "overlay": to_caller_path(str(overlay_path), alias, work_base=work_base),
            "box_format": "xyxy pixels, derived from the mask extent",
            "prompts": prompt_list,
            "id_note": (
                "The number stamped on each colored region of the overlay is "
                "that instance's 0-based index in the masks array (usable as "
                "/measure's instance); its 'label' is the prompt that "
                "produced it."
            ),
            "instances": per_instance,
            **scene_extra,
        },
    )


@app.post("/drop")
async def drop_instances(request: Request) -> JSONResponse:
    """Permanently remove wrong or duplicate instances from a frame's masks.

    Body: {"frame": 2, "instances": [1]} plus optional "image" (that
    frame's picture — when given, a re-rendered overlay is returned) and
    "masks" (explicit artifact path; default newest for the frame).

    Unlike /bev's per-call `erase` (a view-level removal for the motion
    split, which must be repeated on every call), /drop rewrites the masks
    artifact itself: the instance's mask is zeroed IN PLACE — the row and
    every other instance's id stay stable — and the result is saved as a
    NEW masks artifact, which auto-discovery then serves to every later
    /bev, compass grounding, and /measure. Use it when the question
    implies ONE such object ("the desk") but segmentation returned
    several: view the overlay, keep the intended instance, drop the rest.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    try:
        frame = int(payload.get("frame"))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame must be an integer frame index") from exc
    raw = payload.get("instances")
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, list) or not raw or len(raw) > 16:
        raise BadRequest(
            "instances must be a non-empty list (max 16) of overlay ids "
            "(the numbers stamped on the segment overlay)"
        )
    try:
        ids = sorted({int(i) for i in raw})
    except (TypeError, ValueError) as exc:
        raise BadRequest(
            "instances must be integer overlay ids — to pick among "
            "same-label duplicates, ids (not labels) identify which one",
            got=raw,
        ) from exc

    if payload.get("masks"):
        env = load_envelope(payload["masks"], "masks", work_base=work_base)
    else:
        env = _newest_artifact("masks", [frame], "masks", work_base)
    masks = np.array(env["array"], dtype=bool, copy=True)
    if masks.ndim != 3:
        raise BadRequest("masks must be (n, H, W)", shape=list(masks.shape))
    n = int(masks.shape[0])
    labels = list(env.get("labels") or [])

    def _label(i: int) -> str:
        return str(labels[i] if i < len(labels) else "region")

    bad = [i for i in ids if not (0 <= i < n)]
    if bad:
        raise BadRequest(
            f"instance ids {bad} do not exist in frame {frame}",
            available=[f"{i}: {_label(i)}" for i in range(n)],
        )
    already = set(int(i) for i in env.get("dropped_ids") or [])
    dropped_now, notes = [], []
    for i in ids:
        if i in already or not masks[i].any():
            notes.append(f"f{frame}:{i} was already empty")
        else:
            dropped_now.append(i)
        masks[i] = False
    all_dropped = sorted(already | set(ids))
    remaining = [
        {"id": i, "label": _label(i), "mask_pixels": int(masks[i].sum())}
        for i in range(n)
        if masks[i].any()
    ]
    if not remaining:
        notes.append(
            "every instance of this frame is now dropped; later /bev calls "
            "will treat the frame as having no labeled objects"
        )

    mask_path = save_array(
        masks,
        "segmasks",
        [frame],
        "masks",
        extra={
            "layout": "(n_instances, H, W) boolean instance masks",
            "source": "drop",
            "prompts": env.get("prompts"),
            "labels": [_label(i) for i in range(n)],
            "image": env.get("image"),
            # zeroed rows keep every surviving id stable; this records which
            "dropped_ids": all_dropped,
        },
        work_base=work_base,
    )

    overlay_reply = {}
    if payload.get("image"):
        image_path = resolve_in_work_dir(
            payload["image"], must_exist=True, field="image", work_base=work_base
        )
        overlay_img = _segment_overlay(load_image(image_path), masks)
        with artifact_writer(
            "segment_overlay", ".png", subdir="renders", work_base=work_base
        ) as writer:
            overlay_path = writer.path
            with writer.open() as handle:
                overlay_img.save(handle, format="PNG")
        overlay_reply = {
            "overlay": to_caller_path(str(overlay_path), alias, work_base=work_base)
        }

    # Scene-conditional (new configurations only): the instance registry is derived
    # from the newest masks, so the drop is already reflected — report the
    # refreshed registry so the model sees which names survived.
    scene_extra: Dict[str, Any] = {}
    scene = _scene_for(work_base)
    if scene is not None:
        scene_extra = _drop_scene_extra(scene, work_base)

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "frame": frame,
            "dropped": [f"f{frame}:{i} {_label(i)}" for i in dropped_now],
            **({"notes": notes} if notes else {}),
            "remaining": remaining,
            "arrays": {"masks": to_caller_path(mask_path, alias, work_base=work_base)},
            **overlay_reply,
            "note": (
                "the dropped instances are gone from every LATER /bev map, "
                "compass grounding, and measurement — ids of the surviving "
                "instances are unchanged. Maps rendered before this call "
                "still show them; re-run bev"
            ),
            **scene_extra,
        },
    )


def _mask_points(
    points_env: Dict[str, Any],
    masks_env: Dict[str, Any],
    frame_index: int,
    instance: int,
    confidence_env: Optional[Dict[str, Any]] = None,
    min_confidence: float = 0.0,
) -> np.ndarray:
    """3D points inside one instance mask on one frame, alignment-checked."""
    envelopes = [points_env, masks_env]
    if confidence_env is not None:
        envelopes.append(confidence_env)
    # Enforced server side: the measured frame must exist in every input array.
    validate_alignment(envelopes, frame_index=frame_index)

    points = points_env["array"]
    masks = masks_env["array"]
    frames = [int(i) for i in points_env["frame_indices"]]
    local = frames.index(int(frame_index))

    if points.ndim != 4 or points.shape[-1] != 3:
        raise BadRequest(
            "points array must have shape (N, H, W, 3)", shape=list(points.shape)
        )
    if masks.ndim != 3:
        raise BadRequest(
            "masks array must have shape (n_instances, H, W)", shape=list(masks.shape)
        )
    if instance < 0 or instance >= masks.shape[0]:
        raise BadRequest(
            f"instance {instance} out of range", n_instances=int(masks.shape[0])
        )

    frame_points = points[local]
    mask = masks[instance].astype(bool)
    if mask.shape != frame_points.shape[:2]:
        raise BadRequest(
            "mask resolution does not match the point map; both must describe "
            "the same image",
            mask_shape=list(mask.shape),
            point_map_shape=list(frame_points.shape[:2]),
        )

    selected = frame_points[mask]
    keep = np.isfinite(selected).all(axis=1)
    if confidence_env is not None and min_confidence > 0:
        conf = confidence_env["array"][local][mask]
        keep &= conf >= min_confidence
    return selected[keep]


def _rigid_icp(A: np.ndarray, B: np.ndarray, iters: int = 120, trim: float = 0.95):
    """Trimmed rigid ICP: R, t with B ~= R@A + t, plus rms and n used.

    Brute-force nearest neighbours (inputs are subsampled to ~1.5k points)
    with the worst 5% of pairs trimmed each round — on a validated pan
    cloud this recovers injected 10/20/40 deg yaws exactly, where an 80%
    trim slid on the rotation-symmetric body and under-read by 15%.
    """
    R = np.eye(3)
    t = B.mean(0) - A.mean(0)
    for _ in range(iters):
        P = (R @ A.T).T + t
        d2 = ((P[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(P)), j])
        k = dist <= np.quantile(dist, trim)
        Pa, Bb = A[k], B[j[k]]
        ca, cb = Pa.mean(0), Bb.mean(0)
        H = (Pa - ca).T @ (Bb - cb)
        U, _s, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
        R_new = Vt.T @ D @ U.T
        t_new = cb - R_new @ ca
        done = np.allclose(R_new, R, atol=1e-7) and np.allclose(t_new, t, atol=1e-7)
        R, t = R_new, t_new
        if done:
            break
    P = (R @ A.T).T + t
    d2 = ((P[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(d2.min(1))
    k = dist <= np.quantile(dist, trim)
    rms = float(np.sqrt((dist[k] ** 2).mean())) if k.any() else float("nan")
    return R, t, rms, int(k.sum())


@app.post("/motion")
async def object_motion(request: Request) -> JSONResponse:
    """Rigid motion of ONE object between two frames' reconstructions.

    Body: {"frame": <reference frame for axes>, "from": "f<N>:<label-or-id>",
    "to": "f<M>:<label-or-id>"}. Fits a trimmed rigid transform between
    the object's reconstructed points in the two frames, expressed in the
    reference frame's gravity-aligned ground basis: translation
    (right/forward/up metres), yaw about vertical (+ = clockwise seen
    from above), and tilt off vertical with the side that dipped. The
    reply's note carries the honest limits (dark/textureless surfaces can
    reconstruct as static).

    Scene grammar (new configurations): {"of": [A, B]} — two cameras give the
    relative pose between those instants; the same object referenced in
    two frames (f<i>:X, f<j>:X) gives its shift and rotation. Handled by
    _scene_motion; the legacy body above stays byte-identical for old
    configurations.
    """
    payload = await read_json_body(request)
    if payload.get("of") is not None:
        return await _scene_motion(payload)
    work_base = resolve_work_base(payload)
    try:
        frame = int(payload.get("frame"))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame must be an integer frame index") from exc
    refs = {}
    for side in ("from", "to"):
        parsed = _parse_compass_ref(payload.get(side), f"motion.{side}")
        if parsed[0] != "object":
            raise BadRequest(
                f'motion.{side} must be "f<N>:<label-or-id>" (a segmented '
                "instance) — motion is between two views of an OBJECT",
                got=str(payload.get(side)),
            )
        refs[side] = parsed[1]
    (fa, sel_a), (fb, sel_b) = refs["from"], refs["to"]
    if fa == fb:
        raise BadRequest(
            "motion.from and motion.to must name DIFFERENT frames — the "
            "motion is the object's change between two moments"
        )

    points_env = _newest_artifact(
        "points", sorted({fa, fb, frame}), "points", work_base, contains=True
    )
    poses_env = _newest_artifact(
        "camera_poses", [frame], "camera_poses", work_base, contains=True
    )
    gravity_env = _newest_artifact("gravity", [frame], "gravity", work_base)
    points = points_env["array"]
    points_frames = [int(i) for i in points_env["frame_indices"]]
    poses_frames = [int(i) for i in poses_env["frame_indices"]]
    pose = np.asarray(poses_env["array"][poses_frames.index(frame)], dtype=np.float64)
    rot_c2w, cam_pos = pose[:3, :3], pose[:3, 3]
    g = np.asarray(gravity_env["array"], dtype=np.float64).ravel()
    if g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) < 1e-6:
        raise BadRequest("gravity artifact must be a finite (3,) vector")
    down = g / np.linalg.norm(g)
    fwd = np.array([0.0, 0.0, 1.0]) - down * down[2]
    if np.linalg.norm(fwd) < 0.2:
        fwd = np.array([0.0, -1.0, 0.0]) - down * -down[1]
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(down, fwd)
    right = right / np.linalg.norm(right)
    crop = points_env.get("crop_box") or {}
    left_off, top_off = int(crop.get("left", 0)), int(crop.get("top", 0))
    grid_h, grid_w = int(points.shape[1]), int(points.shape[2])

    def _cloud(fi: int, selector: str, side: str) -> np.ndarray:
        env = _newest_artifact("masks", [fi], "masks", work_base)
        masks = env["array"]
        labels = list(env.get("labels") or [])
        dropped = {int(i) for i in env.get("dropped_ids") or []}
        rows = []
        for i in range(int(masks.shape[0])):
            if i in dropped:
                continue
            label = str(labels[i] if i < len(labels) else "region")
            if selector == str(i) or selector.lower() == label.lower():
                rows.append((i, label))
        if not rows:
            raise BadRequest(
                f"motion.{side} f{fi}:{selector} matched no instance",
                available=[
                    f"f{fi}:{i} {labels[i] if i < len(labels) else 'region'}"
                    for i in range(int(masks.shape[0]))
                    if i not in dropped
                ],
            )
        pts = []
        lf = points_frames.index(fi)
        for i, _label in rows:
            ys, xs = np.nonzero(masks[i])
            xc, yc = xs - left_off, ys - top_off
            ok = (xc >= 0) & (xc < grid_w) & (yc >= 0) & (yc < grid_h)
            pts.append(points[lf][yc[ok], xc[ok]].astype(np.float64))
        pw = np.concatenate(pts)
        pw = pw[np.isfinite(pw).all(axis=1)]
        if len(pw) > 1500:
            pw = pw[:: len(pw) // 1500 + 1]
        if len(pw) < 50:
            raise BadRequest(
                f"motion.{side} f{fi}:{selector}: only {len(pw)} usable 3D "
                "points — too few to fit a rigid motion"
            )
        pc = (pw - cam_pos) @ rot_c2w
        return np.stack([pc @ right, pc @ fwd, -(pc @ down)], axis=1)

    A = _cloud(fa, sel_a, "from")
    B = _cloud(fb, sel_b, "to")
    R, t, rms, n_used = _rigid_icp(A, B)
    dtr = B.mean(0) - A.mean(0)
    du, dv, dh = (float(x) for x in dtr)
    dist = math.hypot(du, dv)
    e = R @ np.array([0.0, 1.0, 0.0])
    yaw = math.degrees(math.atan2(float(e[0]), float(e[1])))
    w = R @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(float(np.clip(w[2], -1.0, 1.0))))
    # a surface sloping down toward +x has normal (tan, 0, 1): the normal
    # leans TOWARD the dipping side, so the dip bearing is +(w_u, w_v)
    dip_az = math.degrees(math.atan2(float(w[0]), float(w[1]))) % 360.0
    moved = dist >= 0.10 or abs(dh) >= 0.10
    turned = abs(yaw) >= 8.0
    tilted = tilt >= 10.0
    parts = []
    if moved:
        parts.append(
            f"moved {dist:.2f} m on the ground (right {du:+.2f}, forward "
            f"{dv:+.2f}) and {dh:+.2f} m vertically"
        )
    if turned:
        parts.append(
            f"rotated {yaw:+.1f} deg about vertical "
            f"({'clockwise' if yaw > 0 else 'counterclockwise'} seen from above)"
        )
    if tilted:
        parts.append(
            f"tilted {tilt:.1f} deg off vertical, dipping toward map bearing "
            f"{dip_az:.0f}"
        )
    conclusion = (
        f"f{fa}:{sel_a} -> f{fb}:{sel_b} " + "; ".join(parts)
        if parts
        else (
            f"no significant motion detected in the reconstruction "
            f"(moved {dist:.2f} m, yaw {yaw:+.1f} deg, tilt {tilt:.1f} deg)"
        )
    )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "from": f"f{fa}:{sel_a}",
            "to": f"f{fb}:{sel_b}",
            "translation_m": {
                "right": round(du, 2),
                "forward": round(dv, 2),
                "up": round(dh, 2),
            },
            "ground_distance_m": round(dist, 2),
            "yaw_deg": round(yaw, 1) + 0.0,
            "tilt_deg": round(tilt, 1),
            "tilt_dip_bearing_deg": round(dip_az, 1) if tilt >= 3.0 else None,
            "residual_rms_m": round(rms, 3),
            "n_points": {"from": int(len(A)), "to": int(len(B))},
            "conclusion": conclusion,
            "note": (
                "rigid fit between the two frames' point clouds, in the "
                "reference frame's gravity-aligned ground basis; + yaw = "
                "clockwise seen from above; bearings are clockwise from "
                "map-up like heading_deg. HONEST LIMIT: depth models often "
                "reconstruct dark or textureless objects as static — when "
                "these numbers say static but the photos clearly show the "
                "object being held or manipulated, judge the motion from "
                "the photos instead (subtract the cameras' roll_deg when "
                "comparing angles)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# /match — LoMa-B pairwise keypoint matching (seg-align v4 protocol, step 1)
# ---------------------------------------------------------------------------


@app.post("/match")
async def match(request: Request) -> JSONResponse:
    """LoMa-B keypoint matches for every frame pair, saved to disk.

    Body: {"images": ["frame_0.png", ...], "frame_indices": [0, ...],
           "num_keypoints": optional}

    For each unordered pair (i, j), i < j, runs the upstream LoMa-B matcher
    and saves one (N, 4) float32 array [xA, yA, xB, yB] (pixel coordinates on
    each frame's ORIGINAL grid) with a frame-identity sidecar naming
    frame_indices=[i, j]. /egomotion consumes these — by explicit path or by
    newest-for-that-pair auto-discovery.

    All pairs, not just adjacent ones, because the protocol says the question
    may relate any two views; MMSI tops out at 10 frames = 45 pairs.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)

    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or len(raw_images) < 2:
        raise BadRequest("images must be a list of at least two frame paths")
    if len(raw_images) > 12:
        raise BadRequest("at most 12 images per call", got=len(raw_images))
    paths = [
        resolve_in_work_dir(p, must_exist=True, field="images", work_base=work_base)
        for p in raw_images
    ]
    frame_indices = payload.get("frame_indices") or list(range(len(paths)))
    try:
        frame_indices = [int(i) for i in frame_indices]
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame_indices must be integers") from exc
    if len(frame_indices) != len(paths):
        raise BadRequest(
            "frame_indices must have one entry per image",
            images=len(paths),
            frame_indices=frame_indices,
        )
    if len(set(frame_indices)) != len(frame_indices):
        raise BadRequest("frame_indices must be unique", frame_indices=frame_indices)

    kwargs_extra: Dict[str, Any] = {}
    if payload.get("num_keypoints") is not None:
        try:
            num_keypoints = int(payload["num_keypoints"])
        except (TypeError, ValueError) as exc:
            raise BadRequest("num_keypoints must be an integer") from exc
        if not (64 <= num_keypoints <= 8192):
            raise BadRequest("num_keypoints must be in 64..8192", got=num_keypoints)
        kwargs_extra["num_keypoints"] = num_keypoints

    images = [load_image(p) for p in paths]
    order = sorted(range(len(paths)), key=lambda k: frame_indices[k])
    pairs = []
    for ai in range(len(order)):
        for bi in range(ai + 1, len(order)):
            a, b = order[ai], order[bi]
            result = await call_upstream_async(
                "spatial_Match",
                "match",
                {"image_a": images[a], "image_b": images[b], **kwargs_extra},
            )
            kpts_a = as_numpy(
                result.get("kpts_a") if isinstance(result, dict) else None
            )
            kpts_b = as_numpy(
                result.get("kpts_b") if isinstance(result, dict) else None
            )
            if kpts_a is None or kpts_b is None or kpts_a.shape != kpts_b.shape:
                raise ToolFailure("matcher returned no aligned keypoint arrays")
            stacked = np.concatenate(
                [kpts_a.astype(np.float32), kpts_b.astype(np.float32)], axis=1
            )
            fa, fb = frame_indices[a], frame_indices[b]
            path = save_array(
                stacked,
                "matches",
                [fa, fb],
                "matches",
                extra={
                    "layout": "(n_matches, 4) float32 [xA, yA, xB, yB] pixel coords",
                    "matcher": "loma_B",
                    "size_a": [images[a].width, images[a].height],
                    "size_b": [images[b].width, images[b].height],
                    "image_a": str(paths[a]),
                    "image_b": str(paths[b]),
                },
                work_base=work_base,
            )
            pairs.append(
                {
                    "frames": [fa, fb],
                    "n_matches": int(stacked.shape[0]),
                    "matches": to_caller_path(path, alias, work_base=work_base),
                }
            )

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "matcher": "loma_B",
            "n_pairs": len(pairs),
            "pairs": pairs,
            "note": (
                "Matched keypoints are saved per pair; row i of each array is "
                "one correspondence [xA, yA, xB, yB]. /egomotion picks these "
                "up automatically for a given pair."
            ),
        },
    )


# ---------------------------------------------------------------------------
# /egomotion — camera R, t between two frames from static keypoints
# (fundamental matrix + decomposition; seg-align v4 protocol, step 4)
# ---------------------------------------------------------------------------

#: COLMAP-style focal prior for uncalibrated images. MMSI frames carry no EXIF
#: intrinsics; a wrong-but-consistent prior biases the absolute angles a few
#: degrees while leaving the left/right/forward classification stable, which is
#: what the protocol consumes.
_FOCAL_PRIOR = 1.2


def _assumed_intrinsics(width: int, height: int) -> np.ndarray:
    f = _FOCAL_PRIOR * max(width, height)
    return np.array(
        [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _yaw_pitch_roll(rotation: np.ndarray) -> Dict[str, float]:
    """Decompose a camera-B-in-camera-A rotation into yaw/pitch/roll degrees.

    OpenCV camera axes: +x right, +y down, +z forward. Decomposition order
    R = Ry(yaw) @ Rx(pitch) @ Rz(roll), i.e. yaw about the vertical axis first.
    Signs are chosen for the human reading (locked by unit tests on synthetic
    rotations): yaw > 0 = camera turned right, pitch > 0 = tilted up,
    roll > 0 = tilted clockwise (as seen by the camera).
    """
    fwd = rotation @ np.array([0.0, 0.0, 1.0])
    yaw = math.degrees(math.atan2(float(fwd[0]), float(fwd[2])))
    pitch = math.degrees(-math.asin(max(-1.0, min(1.0, float(fwd[1])))))
    # Undo yaw and pitch, then read roll off the residual rotation about z.
    ry = np.array(
        [
            [math.cos(math.radians(-yaw)), 0.0, math.sin(math.radians(-yaw))],
            [0.0, 1.0, 0.0],
            [-math.sin(math.radians(-yaw)), 0.0, math.cos(math.radians(-yaw))],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(math.radians(-pitch)), math.sin(math.radians(-pitch))],
            [0.0, -math.sin(math.radians(-pitch)), math.cos(math.radians(-pitch))],
        ]
    )
    residual = rx @ ry @ rotation
    # +y is DOWN, so a clockwise roll (the right vector swinging toward +y,
    # exactly how clock hands move when the image is viewed normally) has
    # residual[1, 0] > 0 — plain atan2 already reads clockwise as positive.
    roll = math.degrees(math.atan2(float(residual[1, 0]), float(residual[1, 1])))
    return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}


def _motion_summary(angles: Dict[str, float], direction: np.ndarray) -> str:
    """One plain-language sentence for the VLM to reason with."""
    parts = []
    yaw, pitch = angles["yaw_deg"], angles["pitch_deg"]
    if abs(yaw) >= 3.0:
        parts.append(f"turned {'right' if yaw > 0 else 'left'} by ~{abs(yaw):.0f}°")
    if abs(pitch) >= 3.0:
        parts.append(f"tilted {'up' if pitch > 0 else 'down'} by ~{abs(pitch):.0f}°")
    if not parts:
        parts.append("kept nearly the same orientation")
    x, y, z = (float(v) for v in direction)
    moves = []
    if abs(x) >= 0.25:
        moves.append("right" if x > 0 else "left")
    if abs(z) >= 0.25:
        moves.append("forward" if z > 0 else "backward")
    if abs(y) >= 0.25:
        moves.append("down" if y > 0 else "up")
    parts.append("moved " + " and ".join(moves) if moves else "barely changed position")
    return "camera B relative to camera A: " + ", ".join(parts) + "."


#: Which endpoint produces each auto-discoverable artifact kind — used to
#: phrase the "run X first" guidance when discovery comes up empty.
_ARTIFACT_SOURCES = {
    "matches": "/match",
    "masks": "/segment",
    "points": "/reconstruct",
    "depth": "/reconstruct",
    "camera_poses": "/reconstruct",
    "gravity": "/gravity",
    "dynamic": "/recon",
    "dynamic_prob": "/recon",
}


def _newest_artifact(
    kind: str,
    frame_indices: List[int],
    field: str,
    work_base: Optional[Path],
    contains: bool = False,
) -> Dict[str, Any]:
    """Newest saved array of `kind` whose sidecar names these frames.

    With contains=False the sidecar's frame list must match exactly (a mask for
    one frame, matches for one pair); with contains=True it must merely include
    every requested frame (a reconstruction over frames [0..3] serves a /bev of
    frame 2).

    Auto-discovery exists because the v2 smoke showed per-step path plumbing is
    what kills protocol compliance: the newest artifact for the frames is almost
    always the intended one (work_dir is per-trial), and an explicit path
    parameter still overrides when it is not.
    """
    base = work_base if work_base is not None else WORK_DIR
    wanted = [int(i) for i in frame_indices]
    best: Optional[Dict[str, Any]] = None
    for sidecar_path in sorted((base / "artifacts").glob("*.npy.json")):
        try:
            meta = json.loads(sidecar_path.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("kind") != kind:
            continue
        have = [int(i) for i in meta.get("frame_indices", [])]
        if contains:
            if not all(f in have for f in wanted):
                continue
        elif have != wanted:
            continue
        if best is None or str(meta.get("created", "")) > str(best.get("created", "")):
            best = meta
    if best is None:
        raise BadRequest(
            f"no saved {kind} array found for frames {wanted}; run "
            f"{_ARTIFACT_SOURCES.get(kind, '/reconstruct')} first or pass "
            f"{field} explicitly",
            frames=wanted,
        )
    return load_envelope(best["path"], field, work_base=work_base)


def _optional_artifact(
    kind: str,
    frame_indices: List[int],
    field: str,
    work_base: Optional[Path],
    contains: bool = False,
) -> Optional[Dict[str, Any]]:
    """`_newest_artifact`, but absence is None instead of a 400.

    For side-channel artifacts (the Pi3MOS dynamic masks) that a scene may
    legitimately lack: an older scene, or the side-channel was down at
    /recon time. Every consumer must degrade to the pre-dynamic behaviour
    when this returns None.
    """
    try:
        return _newest_artifact(kind, frame_indices, field, work_base, contains=contains)
    except BadRequest:
        return None


#: Hysteresis rule for the Pi3MOS motion probability: a pixel is DYNAMIC
#: when it lies in a >=_DYNAMIC_FLOOR region 4-connected to at least one
#: >=_DYNAMIC_CORE pixel. Cores alone under-cover the mover (its full
#: extent lives near the floor); the floor alone bleeds weak probability
#: over the background — the MMSI-Bench 996/992/981 probe (2026-09-03)
#: picked exactly this pairing.
_DYNAMIC_CORE = 0.4
_DYNAMIC_FLOOR = 0.05
#: An instance is tagged moving when at least this fraction of its mask
#: pixels are dynamic.
_DYNAMIC_INSTANCE_FRAC = 0.5

#: /motion object-pair significance gates. Rotation (yaw AND tilt) reports
#: from 3 deg — MMSI_376 measured a real 7 deg tilt / 3.5 deg yaw that the
#: old 8/10 deg gates swallowed into "no significant motion". The shift
#: gate is a PERCENTAGE of the object's own size (see the use site),
#: clamped between a reconstruction-noise floor and an absolute cap.
_MOTION_ROT_DEG = 3.0
_MOTION_SHIFT_FRAC = 0.15
_MOTION_SHIFT_FLOOR_M = 0.04
_MOTION_SHIFT_CAP_M = 0.50


def _hysteresis_dynamic(prob: np.ndarray, max_iters: int = 4096) -> np.ndarray:
    """(H, W) probability -> bool dynamic mask via core-seeded flood.

    Pure-numpy dilation fixpoint (scipy is not a gateway dependency). The
    iteration cap only guards a pathological snake-shaped domain; every
    real mask converges in tens of rounds.
    """
    domain = prob >= _DYNAMIC_FLOOR
    out = prob >= _DYNAMIC_CORE
    if not out.any():
        return out
    for _ in range(max_iters):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown &= domain
        if (grown == out).all():
            break
        out = grown
    return out


def _dynamic_flat_keep(
    dyn_env: Optional[Dict[str, Any]],
    points_frames: List[int],
    grid_h: int,
    grid_w: int,
    anchor_frame: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Frame-major flat keep mask implementing the scene cloud rule.

    A query anchored on ``anchor_frame`` reads the fused STATIC points of
    every frame plus ALL points of the anchor frame itself (so per-frame
    facts about a mover — its position as seen there, /motion's ICP —
    stay available while other frames' views of it stop polluting the
    fusion). ``anchor_frame=None`` is the global form: static points only.
    None when the scene has no dynamic artifact — callers then keep
    everything, the pre-dynamic behaviour.
    """
    if dyn_env is None:
        return None
    dyn = np.asarray(dyn_env["array"])
    if dyn.ndim != 3 or dyn.shape[1:] != (grid_h, grid_w):
        return None
    dyn_frames = [int(i) for i in dyn_env.get("frame_indices") or []]
    keep = np.ones(len(points_frames) * grid_h * grid_w, dtype=bool)
    for lf, fi in enumerate(points_frames):
        if anchor_frame is not None and int(fi) == int(anchor_frame):
            continue
        if int(fi) not in dyn_frames:
            continue
        block = dyn[dyn_frames.index(int(fi))]
        keep[lf * grid_h * grid_w : (lf + 1) * grid_h * grid_w] = ~block.astype(
            bool
        ).reshape(-1)
    return keep


def _moving_union(
    masks: np.ndarray, moving_ids: List[int], margin_px: int
) -> np.ndarray:
    """Union of the moving instance masks, dilated by a safety margin.

    The margin absorbs mask under-segmentation at object boundaries: a
    keypoint sitting one pixel outside a moving object's mask is far more
    likely to be on that object than on the background.
    """
    union = np.zeros(masks.shape[1:], dtype=bool)
    for i in moving_ids:
        union |= masks[i].astype(bool)
    if margin_px > 0 and union.any():
        kernel = np.ones((2 * margin_px + 1, 2 * margin_px + 1), dtype=np.uint8)
        union = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)
    return union


def _parse_instance_ids(raw: Any, field: str, n_instances: int) -> List[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise BadRequest(f"{field} must be a list of instance ids", got=repr(raw))
    ids = []
    for v in raw:
        try:
            as_int = int(v)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"{field} entries must be integers", got=repr(v)) from exc
        if not (0 <= as_int < n_instances):
            raise BadRequest(
                f"{field} id {as_int} out of range", n_instances=n_instances
            )
        ids.append(as_int)
    return sorted(set(ids))


@app.post("/egomotion")
async def egomotion(request: Request) -> JSONResponse:
    """Pairwise camera rotation and translation direction from static points.

    Body: {"pair": [0, 1], "moving_a": [ids...], "moving_b": [ids...],
           "matches": optional path, "masks_a": optional path,
           "masks_b": optional path, "ransac_thresh_px": optional,
           "moving_margin_px": optional}

    Keypoint matches (from /match) landing inside a moving instance mask
    (from /segment) in EITHER frame are excluded; the fundamental matrix is
    RANSAC-estimated (USAC MAGSAC) from the static remainder, upgraded to an
    essential matrix with an assumed focal prior, and decomposed into the
    camera rotation and a unit translation direction. Pure CPU — no GPU call.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)

    pair = payload.get("pair")
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(v, (int, float)) or int(v) != v for v in pair)
    ):
        raise BadRequest("pair must be [frame_a, frame_b] with two integers")
    fa, fb = int(pair[0]), int(pair[1])
    if fa == fb:
        raise BadRequest("pair must name two different frames", pair=[fa, fb])
    if fa > fb:
        raise BadRequest(
            "pair must be [smaller, larger] to match how /match saves pairs",
            pair=[fa, fb],
        )

    try:
        ransac_thresh = float(payload.get("ransac_thresh_px", 0.75))
    except (TypeError, ValueError) as exc:
        raise BadRequest("ransac_thresh_px must be a number") from exc
    if not (0.05 <= ransac_thresh <= 10.0):
        raise BadRequest("ransac_thresh_px must be in 0.05..10", got=ransac_thresh)
    try:
        margin_px = int(payload.get("moving_margin_px", 3))
    except (TypeError, ValueError) as exc:
        raise BadRequest("moving_margin_px must be an integer") from exc
    if not (0 <= margin_px <= 50):
        raise BadRequest("moving_margin_px must be in 0..50", got=margin_px)

    matches_env = (
        load_envelope(payload["matches"], "matches", work_base=work_base)
        if payload.get("matches")
        else _newest_artifact("matches", [fa, fb], "matches", work_base)
    )
    if [int(i) for i in matches_env.get("frame_indices", [])] != [fa, fb]:
        raise BadRequest(
            "matches array describes different frames than the requested pair",
            matches_frames=matches_env.get("frame_indices"),
            pair=[fa, fb],
        )
    matches = matches_env["array"]
    if matches.ndim != 2 or matches.shape[1] != 4:
        raise BadRequest(
            "matches array must have shape (n, 4) [xA, yA, xB, yB]",
            shape=list(matches.shape),
        )

    masks_env_a = (
        load_envelope(payload["masks_a"], "masks_a", work_base=work_base)
        if payload.get("masks_a")
        else _newest_artifact("masks", [fa], "masks_a", work_base)
    )
    masks_env_b = (
        load_envelope(payload["masks_b"], "masks_b", work_base=work_base)
        if payload.get("masks_b")
        else _newest_artifact("masks", [fb], "masks_b", work_base)
    )
    for env, frame, field in (
        (masks_env_a, fa, "masks_a"),
        (masks_env_b, fb, "masks_b"),
    ):
        if frame not in [int(i) for i in env.get("frame_indices", [])]:
            raise BadRequest(
                f"{field} does not describe frame {frame}",
                masks_frames=env.get("frame_indices"),
            )
        if env["array"].ndim != 3:
            raise BadRequest(
                f"{field} must have shape (n_instances, H, W)",
                shape=list(env["array"].shape),
            )

    masks_a, masks_b = masks_env_a["array"], masks_env_b["array"]
    moving_a = _parse_instance_ids(
        payload.get("moving_a"), "moving_a", int(masks_a.shape[0])
    )
    moving_b = _parse_instance_ids(
        payload.get("moving_b"), "moving_b", int(masks_b.shape[0])
    )

    size_a = matches_env.get("size_a") or [masks_a.shape[2], masks_a.shape[1]]
    size_b = matches_env.get("size_b") or [masks_b.shape[2], masks_b.shape[1]]
    if [masks_a.shape[2], masks_a.shape[1]] != [int(v) for v in size_a]:
        raise BadRequest(
            "masks_a resolution does not match the matched image; masks and "
            "keypoints must describe the same pixel grid",
            masks_hw=list(masks_a.shape[1:]),
            image_wh=size_a,
        )
    if [masks_b.shape[2], masks_b.shape[1]] != [int(v) for v in size_b]:
        raise BadRequest(
            "masks_b resolution does not match the matched image; masks and "
            "keypoints must describe the same pixel grid",
            masks_hw=list(masks_b.shape[1:]),
            image_wh=size_b,
        )

    pts_a = matches[:, 0:2].astype(np.float64)
    pts_b = matches[:, 2:4].astype(np.float64)

    def _inside(points: np.ndarray, union: np.ndarray) -> np.ndarray:
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, union.shape[1] - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, union.shape[0] - 1)
        return union[ys, xs]

    union_a = _moving_union(masks_a, moving_a, margin_px)
    union_b = _moving_union(masks_b, moving_b, margin_px)
    on_moving = _inside(pts_a, union_a) | _inside(pts_b, union_b)
    static_a, static_b = pts_a[~on_moving], pts_b[~on_moving]
    n_static = int(static_a.shape[0])

    if n_static < 8:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "pair": [fa, fb],
                "n_matches": int(matches.shape[0]),
                "n_excluded_moving": int(on_moving.sum()),
                "n_static": n_static,
                "reason": (
                    "fewer than 8 static correspondences survive the moving-"
                    "object filter, so no fundamental matrix can be estimated. "
                    "This is an observation about the scene (or about the "
                    "moving list), not a tool failure."
                ),
            },
        )

    fund, inlier_mask = cv2.findFundamentalMat(
        static_a,
        static_b,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.999999,
        maxIters=10000,
    )
    if fund is None or fund.shape != (3, 3):
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "pair": [fa, fb],
                "n_static": n_static,
                "reason": (
                    "fundamental matrix estimation failed on the static "
                    "correspondences (degenerate configuration, e.g. almost "
                    "no camera motion or a single plane at low parallax)"
                ),
            },
        )
    inliers = (
        inlier_mask.ravel().astype(bool)
        if inlier_mask is not None
        else np.ones(n_static, dtype=bool)
    )

    k_a = _assumed_intrinsics(int(size_a[0]), int(size_a[1]))
    k_b = _assumed_intrinsics(int(size_b[0]), int(size_b[1]))
    essential = k_b.T @ fund @ k_a
    u, _s, vt = np.linalg.svd(essential)
    essential = u @ np.diag([1.0, 1.0, 0.0]) @ vt

    def _normalize(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        return (np.linalg.inv(intrinsics) @ homogeneous.T).T[:, :2]

    norm_a = _normalize(static_a[inliers], k_a)
    norm_b = _normalize(static_b[inliers], k_b)
    n_cheirality, rot, trans, _pose_mask = cv2.recoverPose(
        essential, norm_a, norm_b, np.eye(3)
    )

    # recoverPose convention: X_b = R @ X_a + t. Express everything in camera
    # A's own axes for the reader: B's orientation is R.T, B's position is
    # -R.T @ t (unit scale — two views fix direction, never magnitude).
    rot_b_in_a = rot.T
    direction = -rot.T @ trans.ravel()
    norm = float(np.linalg.norm(direction))
    if norm > 1e-9:
        direction = direction / norm
    angles = _yaw_pitch_roll(rot_b_in_a)

    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "pair": [fa, fb],
            "n_matches": int(matches.shape[0]),
            "n_excluded_moving": int(on_moving.sum()),
            "n_static": n_static,
            "n_inliers": int(inliers.sum()),
            "inlier_ratio": round(float(inliers.mean()), 4),
            "n_cheirality": int(n_cheirality),
            "moving_a": moving_a,
            "moving_b": moving_b,
            "rotation": {
                "matrix_b_in_a": [
                    [round(float(v), 6) for v in row] for row in rot_b_in_a
                ],
                **{k: round(v, 2) for k, v in angles.items()},
            },
            "translation": {
                "direction_xyz_in_a": [round(float(v), 4) for v in direction],
                "note": (
                    "unit direction of camera B's position in camera A's axes "
                    "(+x right, +y down, +z forward); the magnitude is "
                    "unrecoverable from two views"
                ),
            },
            "summary": _motion_summary(angles, direction),
            **(
                {
                    "low_evidence": True,
                    "low_evidence_note": (
                        "fewer than 15 RANSAC inliers support this estimate; "
                        "treat the summary as weak evidence and prefer what "
                        "the images themselves show if they disagree"
                    ),
                }
                if int(inliers.sum()) < 15
                else {}
            ),
            "axes_note": (
                "all quantities are in camera A's coordinate frame: +x right, "
                "+y down, +z forward (OpenCV convention); yaw>0 = turned "
                "right, pitch>0 = tilted up, roll>0 = clockwise"
            ),
            "intrinsics_note": (
                f"intrinsics are assumed (focal = {_FOCAL_PRIOR} × max(W, H), "
                "principal point at the image center); absolute angles carry "
                "a few degrees of prior-induced error, left/right/forward "
                "classifications are robust to it"
            ),
            "artifacts": {
                "matches": to_caller_path(
                    matches_env.get("resolved_path"), alias, work_base=work_base
                ),
                "masks_a": to_caller_path(
                    masks_env_a.get("resolved_path"), alias, work_base=work_base
                ),
                "masks_b": to_caller_path(
                    masks_env_b.get("resolved_path"), alias, work_base=work_base
                ),
            },
        },
    )


# ---------------------------------------------------------------------------
# /gravity — GeoCalib-estimated gravity direction relative to one camera
# (seg-align v5 protocol; feeds /bev's ground-plane alignment)
# ---------------------------------------------------------------------------


def _lean_pitch_from_gravity(g: np.ndarray) -> tuple:
    """(vertical_lean_deg, camera_pitch_deg) from a unit DOWN vector.

    Inverse of the parameterization the VLM estimator used, kept so sidecars
    and /bev's `gravity_used` report stay comparable across the estimator
    swap: lean > 0 means vertical structures' tops lean image-right, pitch > 0
    means the camera looks up. Camera coords x right / y down / z forward;
    a level upright camera (g = (0, 1, 0)) gives (0, 0).
    """
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(g[2])))))
    lean = math.degrees(math.atan2(-float(g[0]), float(g[1])))
    return lean, pitch


def _gravity_confidence(uncertainty_deg: Any) -> str:
    """Bucket GeoCalib's gravity uncertainty into the sidecar's high/medium/low.

    Thresholds are coarse by design: GeoCalib's median error on in-domain
    photos is ~1°, and /bev only degrades visibly past a few degrees of
    ground-plane tilt.
    """
    try:
        u = float(uncertainty_deg)
    except (TypeError, ValueError):
        return "low"
    if not math.isfinite(u):
        return "low"
    if u <= 2.0:
        return "high"
    if u <= 6.0:
        return "medium"
    return "low"


@app.post("/gravity")
async def gravity(request: Request) -> JSONResponse:
    """GeoCalib-estimated gravity direction relative to one frame's camera.

    Body: {"image": "frame_2.png", "frame_index": 2}

    Sends the frame to the GPU server's GeoCalib tool (single-image geometric
    calibration), which returns the unit gravity (DOWN) direction in that
    camera's coordinates plus an uncertainty; saves it as a frame-identity
    artifact for /bev to discover.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    image_path = resolve_in_work_dir(
        payload.get("image"), must_exist=True, field="image", work_base=work_base
    )
    try:
        frame_index = int(payload.get("frame_index", 0))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame_index must be an integer") from exc

    image = load_image(image_path)
    result = await call_upstream_async(
        "spatial_Gravity", "estimate", {"image_source": image}
    )
    raw = result if isinstance(result, dict) else vars(result)
    vector = as_numpy(raw.get("gravity_xyz"))
    if vector is None:
        raise ToolFailure("gravity estimator returned no gravity_xyz")
    vector = np.asarray(vector, dtype=np.float64).ravel()
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or (np.linalg.norm(vector) < 1e-6)
    ):
        raise ToolFailure(
            "gravity estimator returned a degenerate vector", got=vector.tolist()
        )
    vector = vector / np.linalg.norm(vector)
    lean, pitch = _lean_pitch_from_gravity(vector)
    uncertainty = raw.get("gravity_uncertainty_deg")
    # confidence / uncertainty stay in the SIDECAR only: every number the
    # reply exposes becomes licence to override a correct readout
    # (v10/104 cited "gravity confidence", v14/104 dug the 24.75 deg
    # uncertainty out of this very reply as its sixth excuse).
    sidecar_meta = {
        "vertical_lean_deg": round(lean, 1),
        "camera_pitch_deg": round(pitch, 1),
        "confidence": _gravity_confidence(uncertainty),
        "method": "geocalib",
        **(
            {"gravity_uncertainty_deg": round(float(uncertainty), 2)}
            if isinstance(uncertainty, (int, float)) and math.isfinite(uncertainty)
            else {}
        ),
    }

    path = save_array(
        vector,
        "gravity",
        [frame_index],
        "gravity",
        extra={
            "layout": "(3,) unit gravity direction, camera coords x right / y down / z forward",
            **sidecar_meta,
        },
        work_base=work_base,
    )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "frame_index": frame_index,
            "gravity_xyz": [round(float(v), 4) for v in vector],
            "vertical_lean_deg": round(lean, 1),
            "camera_pitch_deg": round(pitch, 1),
            "method": "geocalib",
            "artifact": to_caller_path(path, alias, work_base=work_base),
            "axes_note": (
                "camera coordinates: +x right, +y down, +z forward; an "
                "upright level camera has gravity (0, 1, 0)"
            ),
        },
    )


# ---------------------------------------------------------------------------
# /bev — gravity-aligned bird's-eye-view map of one reference frame
# ---------------------------------------------------------------------------


def _bev_font(size: int):
    from PIL import ImageFont

    for candidate in _SEGMENT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


# -- compass: nothing in a reconstruction knows north. One orientation fact
# from the QUESTION ("the bed in Figure 1 is in the southwest corner") names
# a start anchor, an end anchor, and the compass direction from start to
# end; that single declared arrow pins the whole rose. Anchors are points
# (an instance's ground centroid, a camera position, the scene footprint
# centre) or planes drawn as ground lines (an instance's fitted plane, a
# camera's image plane); a point-to-plane arrow runs along the
# perpendicular, which is exactly what "the side you see" and "the side of
# the room" mean geometrically.

_COMPASS_ORIENTATIONS = [
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
]
_COMPASS_OFFSET_DEG = {name: 45.0 * i for i, name in enumerate(_COMPASS_ORIENTATIONS)}
_COMPASS_SHORT = {
    "north": "N",
    "northeast": "NE",
    "east": "E",
    "southeast": "SE",
    "south": "S",
    "southwest": "SW",
    "west": "W",
    "northwest": "NW",
}
_COMPASS_ALIASES = {short.lower(): name for name, short in _COMPASS_SHORT.items()}


def _parse_compass_orientation(raw: Any) -> str:
    text = str(raw or "").strip().lower().replace("-", "").replace(" ", "")
    text = _COMPASS_ALIASES.get(text, text)
    if text not in _COMPASS_OFFSET_DEG:
        raise BadRequest(
            "compass orientation must be one of "
            + "/".join(_COMPASS_ORIENTATIONS)
            + " (or N/NE/E/SE/S/SW/W/NW)",
            got=str(raw),
        )
    return text


_GROUND_CELL_M = 0.05


def _ground_cells(us: np.ndarray, vs: np.ndarray) -> tuple:
    """Deduplicate ground samples onto a 5 cm grid: one vote per occupied
    cell, however many 3D points stack on it.

    A vertical surface (a bed's headboard, a wall face) drops hundreds of
    points onto the same ground spot, so a centroid over raw points drifts
    toward it and away from the footprint's middle (v12/466: the bed's
    point centroid landed 0.19 m from the scene centre, headboard-heavy).
    A centroid over occupied cells is the centre of the 2D FOOTPRINT — the
    thing the BEV map actually draws."""
    cells = np.unique(
        np.stack(
            [np.floor(us / _GROUND_CELL_M), np.floor(vs / _GROUND_CELL_M)],
            axis=1,
        ),
        axis=0,
    )
    return (
        (cells[:, 0] + 0.5) * _GROUND_CELL_M,
        (cells[:, 1] + 0.5) * _GROUND_CELL_M,
    )


def _parse_compass_ref(raw: Any, field: str) -> tuple:
    """Parse one compass anchor reference (syntax only, no grounding).

    Grammar: "scene" (the fused footprint's centre), "camera:f<N>" (that
    frame's camera position), "camplane:f<N>" (that frame's image plane —
    always a plane), or "f<N>:<label-or-id>" (a segmented instance of that
    frame — the same selector grammar as erase).
    """
    text = str(raw or "").strip()
    low = text.lower()
    if low == "scene":
        return ("scene", None)
    if low.startswith(("camera:", "camplane:")):
        kind, _, tail = low.partition(":")
        tail = tail.strip().removeprefix("f")
        if tail.isdigit():
            return (kind, int(tail))
        raise BadRequest(f'{field} must name the frame as "{kind}:f<N>"', got=text)
    head, sep, selector = text.partition(":")
    head = head.strip().lower().removeprefix("f")
    selector = selector.strip()
    if sep and selector and head.isdigit():
        return ("object", (int(head), selector))
    raise BadRequest(
        f'{field} must be "f<N>:<label-or-id>" (a segmented instance), '
        '"scene", "camera:f<N>", or "camplane:f<N>"',
        got=text,
    )


def _draw_dashed_line(draw, p0, p1, fill, width=2, dash=9, gap=6):
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    t = 0.0
    while t < dist:
        seg = min(dash, dist - t)
        draw.line(
            [(x0 + ux * t, y0 + uy * t), (x0 + ux * (t + seg), y0 + uy * (t + seg))],
            fill=fill,
            width=width,
        )
        t += dash + gap


def _draw_arrow(draw, p0, p1, fill, width=4, head=14):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    draw.line([p0, p1], fill=fill, width=width)
    bx, by = p1[0] - head * ux, p1[1] - head * uy
    draw.polygon(
        [
            p1,
            (bx - 0.5 * head * uy, by + 0.5 * head * ux),
            (bx + 0.5 * head * uy, by - 0.5 * head * ux),
        ],
        fill=fill,
    )


def _draw_compass_rose(draw, cx, cy, radius, north_bearing_deg, font, color):
    """Rose legend on a white backdrop: north is an arrow, N/E/S/W lettered."""
    pad = radius + 14
    draw.ellipse(
        [cx - pad, cy - pad, cx + pad, cy + pad],
        fill=(255, 255, 255),
        outline=(180, 185, 190),
    )
    for name in _COMPASS_ORIENTATIONS:
        b = math.radians((north_bearing_deg + _COMPASS_OFFSET_DEG[name]) % 360.0)
        ux, uy = math.sin(b), -math.cos(b)  # map-up is pixel -y
        cardinal = name in ("north", "east", "south", "west")
        length = radius if cardinal else 0.55 * radius
        tip = (cx + length * ux, cy + length * uy)
        if name == "north":
            _draw_arrow(draw, (cx, cy), tip, color, width=3, head=10)
        else:
            draw.line(
                [(cx, cy), tip],
                fill=color if cardinal else (150, 155, 160),
                width=2 if cardinal else 1,
            )
        if cardinal:
            draw.text(
                (cx + (length + 10) * ux - 4, cy + (length + 10) * uy - 7),
                _COMPASS_SHORT[name],
                fill=color,
                font=font,
            )


@app.post("/bev")
async def bev(request: Request) -> JSONResponse:
    """Top-down (bird's-eye-view) map fused from every reconstructed frame.

    Body: {"frame": 1} plus optional overrides:
        points / camera_poses / masks / gravity — artifact paths (default:
        newest auto-discovered; "masks" overrides the reference frame only),
        max_labels (default 30), min_label_coverage (default 0 — masks are
        prompted question objects now, so every one is kept and labeled),
        merge_radius_m (default 0.75 — same-label instances from
        different frames within this distance merge into one object),
        range_m, ceiling_margin_m (default 1.0 — points this far above the
        reference camera are treated as ceiling and left out of the map),
        erase (list of "f<N>:<label-or-id>" — remove that frame's view of an
        object from the fusion; two maps erasing the object's OTHER frame
        each show it where one frame saw it, and comparing them reads out
        its motion),
        compass ({"start": ..., "end": ..., "orientation": ..., plus
        optional start_plane/end_plane booleans} — declare that the
        direction from start to end is that compass orientation; the map is
        drawn with the anchor arrow and a derived rose, and the
        orientation→vector mapping is saved as a bev_compass artifact. A
        later /bev on the same reference frame redraws the rose from the
        saved artifact automatically).

    All frames in the reconstruction share one world coordinate system, so
    every frame's point map is projected into the REFERENCE frame's
    gravity-aligned ground plane: the map covers what any view saw. The
    reference camera sits at the origin looking "up" the map; every camera is
    drawn as f<N> with a wedge showing the direction it faces (heading_deg in
    the reply). Objects come from every frame's saved masks — /segment is
    prompted-only, so those are exactly the objects the question names —
    each drawn as a dot at its ground-plane centroid plus its label. The
    flattened 2D layout (camera positions + headings, object centroids) is
    also saved as bev_cameras / bev_objects artifacts.
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    try:
        frame = int(payload.get("frame"))
    except (TypeError, ValueError) as exc:
        raise BadRequest("frame must be an integer frame index") from exc

    try:
        max_labels = int(payload.get("max_labels", 30))
        min_cov = float(payload.get("min_label_coverage", 0.0))
        ceiling_margin = float(payload.get("ceiling_margin_m", 1.0))
        range_m = float(payload["range_m"]) if payload.get("range_m") else None
    except (TypeError, ValueError) as exc:
        raise BadRequest(
            "max_labels/min_label_coverage/range_m must be numbers"
        ) from exc
    if not (1 <= max_labels <= 30):
        raise BadRequest("max_labels must be in 1..30", got=max_labels)

    def _env(kind, field, frames, contains=False):
        if payload.get(field):
            return load_envelope(payload[field], field, work_base=work_base)
        return _newest_artifact(kind, frames, field, work_base, contains=contains)

    points_env = _env("points", "points", [frame], contains=True)
    poses_env = _env("camera_poses", "camera_poses", [frame], contains=True)
    # Scene-conditional (new configurations only — a scene exists iff /recon ran):
    # gravity comes from the scene's fused GeoCalib estimate over the
    # best-confidence frames, so no per-frame /gravity call is needed. An
    # explicit gravity path still overrides, and old configurations keep the
    # artifact-discovery path bit for bit.
    scene = _scene_for(work_base)
    if scene is not None and not payload.get("gravity"):
        scene, gravity_env = await _scene_bev_gravity_env(
            scene, work_base, frame, poses_env
        )
    else:
        gravity_env = _env("gravity", "gravity", [frame])

    validate_alignment([points_env], frame_index=frame)
    points_frames = [int(i) for i in points_env["frame_indices"]]
    poses_frames = [int(i) for i in poses_env["frame_indices"]]
    if frame not in poses_frames:
        raise BadRequest(
            "camera_poses does not cover the reference frame",
            poses_frames=poses_frames,
        )
    points = points_env["array"]
    poses = poses_env["array"]
    # Scene-conditional: a recorded scale calibration (from /distance or
    # /depth) rescales the whole geometry once, up front, so every metre
    # this endpoint reports — objects, cameras, the grid — is calibrated.
    scene_scale = scene_store.scale_factor(scene) if scene is not None else 1.0
    if scene_scale != 1.0:
        points = np.asarray(points, dtype=np.float32) * scene_scale
        poses = np.asarray(poses, dtype=np.float64).copy()
        poses[:, :3, 3] *= scene_scale
    g = np.asarray(gravity_env["array"], dtype=np.float64).ravel()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise BadRequest("points must be (N, H, W, 3)", shape=list(points.shape))

    # Masks for EVERY fused frame, not just the reference one: an object the
    # question cares about may only be visible in another view. A frame with
    # no saved masks contributes background points only. The explicit "masks"
    # parameter still overrides — for the reference frame.
    masks_envs: Dict[int, Dict[str, Any]] = {}
    for fi in points_frames:
        if fi == frame and payload.get("masks"):
            masks_envs[fi] = load_envelope(
                payload["masks"], "masks", work_base=work_base
            )
        else:
            try:
                masks_envs[fi] = _newest_artifact("masks", [fi], "masks", work_base)
            except BadRequest:
                continue
    if not masks_envs:
        raise BadRequest(
            "no saved masks found for any fused frame; run /segment first",
            frames=points_frames,
        )
    for fi, env in masks_envs.items():
        if env["array"].ndim != 3:
            raise BadRequest(
                f"masks for frame {fi} must be (n, H, W)",
                shape=list(env["array"].shape),
            )
    if g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) < 1e-6:
        raise BadRequest("gravity artifact must be a finite (3,) vector")

    pose = np.asarray(poses[poses_frames.index(frame)], dtype=np.float64)
    rot_c2w, cam_pos = pose[:3, :3], pose[:3, 3]
    # Fuse EVERY reconstructed frame: the point maps already share one world
    # coordinate system, so projecting them all into the reference camera's
    # ground frame is free — the map covers what any view saw, not just the
    # reference view's own frustum. Flat indexing stays frame-major so a mask
    # pixel of frame lf addresses lf*H*W + y*W + x.
    grid_h, grid_w = int(points.shape[1]), int(points.shape[2])
    pts_world = points.reshape(-1, 3).astype(np.float64)
    finite = np.isfinite(pts_world).all(axis=1)
    pts_cam = (pts_world - cam_pos) @ rot_c2w  # == R^T (X - t) rowwise

    # Ground-plane basis in the reference camera's own coordinates.
    down = g / np.linalg.norm(g)
    fwd = np.array([0.0, 0.0, 1.0]) - down * down[2]
    if np.linalg.norm(fwd) < 0.2:
        # Camera looking almost straight down: its z-axis is useless as a
        # heading, fall back to its image-up direction.
        fwd = np.array([0.0, -1.0, 0.0]) - down * -down[1]
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(down, fwd)
    right = right / np.linalg.norm(right)

    u_all = pts_cam @ right
    v_all = pts_cam @ fwd
    h_all = -(pts_cam @ down)  # height above the camera, along real up
    keep = finite & (h_all < ceiling_margin)
    # Scene cloud rule: the map fuses every frame's STATIC points plus ALL
    # points of the reference frame. Other frames' views of a moving object
    # leave the fusion (no more two-dots-no-time-order ghosting: each map
    # shows a mover where its own reference frame saw it), while the
    # reference frame keeps its movers so this map stays a complete record
    # of that moment. No dynamic artifact -> no filtering (older scenes).
    dyn_keep = _dynamic_flat_keep(
        _optional_artifact("dynamic", points_frames, "dynamic", work_base, contains=True),
        points_frames,
        grid_h,
        grid_w,
        anchor_frame=frame,
    )
    if dyn_keep is not None:
        keep &= dyn_keep
    if int(keep.sum()) < 100:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "frame": frame,
                "reason": (
                    "fewer than 100 usable 3D points across the fused frames "
                    "after the ceiling filter; the reconstruction cannot "
                    "support a map"
                ),
            },
        )

    # Per-instance BEV positions, on the (possibly cropped) point grid.
    crop = points_env.get("crop_box") or {}
    left_off, top_off = int(crop.get("left", 0)), int(crop.get("top", 0))
    try:
        merge_radius = float(payload.get("merge_radius_m", 0.75))
    except (TypeError, ValueError) as exc:
        raise BadRequest("merge_radius_m must be a number") from exc

    # "erase": ["f1:pan"] removes that frame's matching instance(s) — their
    # POINTS leave the fused cloud and they are never labeled. Two maps that
    # differ only in which frame's view of a moving object was erased show
    # the object at its remaining frame's position each: comparing them IS
    # the motion readout (a flattened map cannot show motion otherwise —
    # both positions land on one map as two dots with no time order).
    raw_erase = payload.get("erase")
    if raw_erase is None:
        raw_erase = []
    if isinstance(raw_erase, str):
        raw_erase = [raw_erase]
    if not isinstance(raw_erase, list) or len(raw_erase) > 16:
        raise BadRequest(
            'erase must be a list (max 16) of "f<N>:<label-or-id>" strings'
        )
    erase_specs = []
    for item in raw_erase:
        text = str(item).strip()
        head, sep, selector = text.partition(":")
        head = head.strip().lower().removeprefix("f")
        selector = selector.strip()
        if not sep or not selector or not head.isdigit():
            raise BadRequest(
                'each erase entry must be "f<N>:<label-or-id>" — the frame '
                "whose view to erase, then that frame's instance label or "
                'overlay id (e.g. "f1:pan" or "f1:0")',
                got=text,
            )
        erase_specs.append((int(head), selector, text))

    # "compass": syntax is checked up front so a malformed request fails
    # before any heavy work; the anchors ground onto the map further down,
    # once the instance candidates and camera positions exist.
    compass_req = payload.get("compass")
    c_orient = None
    c_start_ref = c_end_ref = None
    c_start_plane = c_end_plane = False
    c_queries: List[Dict[str, Any]] = []
    if compass_req is not None:
        if not isinstance(compass_req, dict):
            raise BadRequest(
                'compass must be an object: {"start": ..., "end": ..., '
                '"orientation": ...} and/or {"queries": [...]}'
            )

        def _check_plane_pair(a_ref, a_plane, b_ref, b_plane, where):
            for ref, flag, side in ((a_ref, a_plane, "start"), (b_ref, b_plane, "end")):
                if flag and ref[0] != "object":
                    raise BadRequest(
                        f"{where}.{side}_plane applies to object references "
                        "only — camplane:f<N> is already a plane, and scene "
                        "/ camera:f<N> are points"
                    )
            if (a_ref[0] == "camplane" or a_plane) and (
                b_ref[0] == "camplane" or b_plane
            ):
                raise BadRequest(
                    f"{where}: start and end cannot both be planes; at least "
                    "one side must be a point (a point-to-plane arrow runs "
                    "along the perpendicular)"
                )

        has_anchor_fields = any(
            compass_req.get(k) is not None for k in ("orientation", "start", "end")
        )
        if has_anchor_fields:
            c_orient = _parse_compass_orientation(compass_req.get("orientation"))
            c_start_ref = _parse_compass_ref(compass_req.get("start"), "compass.start")
            c_end_ref = _parse_compass_ref(compass_req.get("end"), "compass.end")
            c_start_plane = bool(compass_req.get("start_plane"))
            c_end_plane = bool(compass_req.get("end_plane"))
            _check_plane_pair(
                c_start_ref, c_start_plane, c_end_ref, c_end_plane, "compass"
            )

        # "queries": read directions OUT of the compass — same anchor grammar,
        # no orientation. Each is answered with a conclusion naming the
        # nearest compass direction (the tool's answer, not a menu) plus the
        # full eight-way ranking, against the north declared in this call or
        # restored from the saved bev_compass artifact.
        raw_queries = compass_req.get("queries")
        if raw_queries is not None:
            if not isinstance(raw_queries, list) or len(raw_queries) > 8:
                raise BadRequest(
                    'compass.queries must be a list (max 8) of {"start": ..., '
                    '"end": ...} objects'
                )
            for k, rq in enumerate(raw_queries):
                if not isinstance(rq, dict):
                    raise BadRequest(f"compass.queries[{k}] must be an object")
                q_start = _parse_compass_ref(
                    rq.get("start"), f"compass.queries[{k}].start"
                )
                q_end = _parse_compass_ref(rq.get("end"), f"compass.queries[{k}].end")
                q_sp = bool(rq.get("start_plane"))
                q_ep = bool(rq.get("end_plane"))
                _check_plane_pair(q_start, q_sp, q_end, q_ep, f"compass.queries[{k}]")
                c_queries.append(
                    {
                        "start_ref": q_start,
                        "end_ref": q_end,
                        "start_raw": str(rq.get("start")),
                        "end_raw": str(rq.get("end")),
                        "start_plane": q_sp,
                        "end_plane": q_ep,
                    }
                )
        if not has_anchor_fields and not c_queries:
            raise BadRequest(
                "compass needs an anchor (start/end/orientation) and/or "
                "queries — an empty compass object does nothing"
            )

    erased_ids: set = set()
    erased_resolved = []
    erased_frame_labels: set = set()
    for eframe, selector, spec_text in erase_specs:
        env = masks_envs.get(eframe)
        if env is None or eframe not in points_frames:
            raise BadRequest(
                f"erase target {spec_text!r}: frame {eframe} has no saved "
                "masks among the fused frames",
                frames_with_masks=sorted(masks_envs),
            )
        frame_masks = env["array"]
        labels = list(env.get("labels") or [])
        matched = []
        for i in range(int(frame_masks.shape[0])):
            label = str(labels[i] if i < len(labels) else "region")
            if selector == str(i) or selector.lower() == label.lower():
                matched.append((i, label))
        if not matched:
            raise BadRequest(
                f"erase target {spec_text!r} matched nothing in frame {eframe}",
                available=[
                    f"{i}: {labels[i] if i < len(labels) else 'region'}"
                    for i in range(int(frame_masks.shape[0]))
                ],
            )
        lf = points_frames.index(eframe)
        base = lf * grid_h * grid_w
        for i, label in matched:
            ys, xs = np.nonzero(frame_masks[i])
            xc, yc = xs - left_off, ys - top_off
            ok_px = (xc >= 0) & (xc < grid_w) & (yc >= 0) & (yc < grid_h)
            keep[base + yc[ok_px] * grid_w + xc[ok_px]] = False
            erased_ids.add((eframe, i))
            erased_resolved.append(f"f{eframe}:{i} {label}")
            erased_frame_labels.add((eframe, label.strip().lower()))

    candidates, skipped = [], []
    for lf, fi in enumerate(points_frames):
        env = masks_envs.get(fi)
        if env is None:
            continue
        frame_masks = env["array"]
        labels = list(env.get("labels") or [])
        # instances removed for good by /drop: their rows are zeroed for id
        # stability — skip them outright so they are neither labeled nor
        # reported as "unlabeled"
        dropped = {int(i) for i in env.get("dropped_ids") or []}
        coverage = frame_masks.reshape(frame_masks.shape[0], -1).mean(axis=1)
        base = lf * grid_h * grid_w
        for i in range(int(frame_masks.shape[0])):
            if (fi, i) in erased_ids or i in dropped:
                continue
            if coverage[i] < min_cov:
                continue
            ys, xs = np.nonzero(frame_masks[i])
            xc, yc = xs - left_off, ys - top_off
            ok_px = (xc >= 0) & (xc < grid_w) & (yc >= 0) & (yc < grid_h)
            if ok_px.sum() == 0:
                skipped.append(f"f{fi}:{i}")
                continue
            flat = base + yc[ok_px] * grid_w + xc[ok_px]
            if flat.size > 4000:
                flat = flat[:: flat.size // 4000 + 1]
            flat = flat[keep[flat]]
            if flat.size < 20:
                skipped.append(f"f{fi}:{i}")
                continue
            cell_u, cell_v = _ground_cells(u_all[flat], v_all[flat])
            candidates.append(
                {
                    "frame": int(fi),
                    "id": int(i),
                    "label": str(labels[i] if i < len(labels) else "region"),
                    "cov": float(coverage[i]),
                    "flat": flat,
                    "u": float(np.median(cell_u)),
                    "v": float(np.median(cell_v)),
                }
            )

    # An erase that leaves a same-label instance of the SAME frame on the
    # map is almost never what the caller meant: split-pair questions speak
    # of ONE object ("the pan"), and a leftover twin keeps that object on
    # both split maps, making them near-identical (v11/MMSI_376: two pans
    # per frame — one in a hand, one resting — erased by id). Warn instead
    # of rendering silently.
    erase_leftovers = [
        f"f{c['frame']}:{c['id']} {c['label']}"
        for c in candidates
        if (c["frame"], c["label"].strip().lower()) in erased_frame_labels
    ]

    # Same physical object seen from several frames: merge instances that
    # share a label and land within merge_radius of each other on the map,
    # so one object gets one name instead of one per view.
    groups: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: -c["cov"]):
        key = cand["label"].strip().lower()
        placed = False
        for grp in groups:
            if grp["key"] == key and (
                math.hypot(cand["u"] - grp["u"], cand["v"] - grp["v"]) <= merge_radius
            ):
                grp["members"].append(cand)
                placed = True
                break
        if not placed:
            groups.append(
                {"key": key, "u": cand["u"], "v": cand["v"], "members": [cand]}
            )
    groups = groups[:max_labels]
    for grp in groups[:]:
        # anchor: the reference frame's view when it has one, else the best
        ref_members = [m for m in grp["members"] if m["frame"] == frame]
        grp["anchor"] = (ref_members or grp["members"])[0]

    objects = []
    inst_points: Dict[int, np.ndarray] = {}
    axis_of: Dict[int, Dict[str, Any]] = {}
    for k, grp in enumerate(groups):
        anchor = grp["anchor"]
        flat = np.concatenate([m["flat"] for m in grp["members"]])
        inst_points[k] = flat
        cell_u, cell_v = _ground_cells(u_all[flat], v_all[flat])
        cu, cv = float(np.median(cell_u)), float(np.median(cell_v))
        # Footprint long axis: the ground bearing (mod 180) of the cell
        # PCA's major axis. This is the tool-side angle readout the split
        # pair needs — v12/376 had the rotation written in the two maps
        # (handle 172°→190° = clockwise) and the model read it backwards
        # by eye. Null when the footprint is too small or too round for
        # the axis to mean anything.
        axis_deg = None
        if cell_u.size >= 30:
            du_c, dv_c = cell_u - cell_u.mean(), cell_v - cell_v.mean()
            cov = np.array(
                [
                    [float(np.mean(du_c * du_c)), float(np.mean(du_c * dv_c))],
                    [float(np.mean(du_c * dv_c)), float(np.mean(dv_c * dv_c))],
                ]
            )
            evals, evecs = np.linalg.eigh(cov)
            minor = math.sqrt(max(float(evals[0]), 1e-12))
            major = math.sqrt(max(float(evals[1]), 0.0))
            if major > 0.05 and major / minor >= 1.15:
                ax = evecs[:, 1]
                axis_deg = math.degrees(math.atan2(ax[0], ax[1])) % 180.0
                t = du_c * ax[0] + dv_c * ax[1]
                axis_of[k] = {
                    "p": np.array([float(cell_u.mean()), float(cell_v.mean())]),
                    "d": ax,
                    "t_lo": float(np.percentile(t, 2.5)),
                    "t_hi": float(np.percentile(t, 97.5)),
                }
        objects.append(
            {
                "label": anchor["label"],
                "shown_as": f"f{anchor['frame']}:{anchor['id']}",
                "instances": sorted(f"f{m['frame']}:{m['id']}" for m in grp["members"]),
                "n_views": len({m["frame"] for m in grp["members"]}),
                "right_m": round(cu, 2),
                "forward_m": round(cv, 2),
                "height_rel_camera_m": round(float(np.median(h_all[flat])), 2),
                "distance_m": round(float(math.hypot(cu, cv)), 2),
                "footprint_axis_deg": (
                    round(axis_deg, 1) if axis_deg is not None else None
                ),
            }
        )

    # Every camera (position AND facing direction) in the reference camera's
    # ground frame. The facing direction is the camera's optical axis (+z)
    # projected onto the ground plane; cam_dirs keeps the unit (u, v) for
    # drawing, heading_deg reports it (0 = the way the reference camera
    # faces, i.e. map-up; 90 = its right). None when the camera looks nearly
    # straight up/down — its heading is then meaningless on the map.
    cameras = []
    cam_dirs: List[Optional[tuple]] = []
    # unrounded ground positions/headings per frame, for compass grounding
    cam_ground: Dict[int, np.ndarray] = {}
    cam_dir_of: Dict[int, Optional[tuple]] = {}
    for k, fi in enumerate(poses_frames):
        pose_k = np.asarray(poses[k], dtype=np.float64)
        rel = (pose_k[:3, 3] - cam_pos) @ rot_c2w
        optical_axis = pose_k[:3, :3] @ np.array([0.0, 0.0, 1.0])
        dir_ref = optical_axis @ rot_c2w  # world direction -> reference cam coords
        hu, hv = float(dir_ref @ right), float(dir_ref @ fwd)
        norm = math.hypot(hu, hv)
        cam_ground[fi] = np.array([float(rel @ right), float(rel @ fwd)])
        cam_dir_of[fi] = (hu / norm, hv / norm) if norm > 0.15 else None
        # In-image roll: the angle gravity makes with image-down in THIS
        # camera's picture (+ = camera leans right). Two photos of the
        # same object can show it apparently rotated purely because the
        # camera rolled between them (v12/376: a +7.5 deg roll change
        # accounted for the pan handle's whole apparent swing) — comparing
        # angles across photos is only valid after subtracting roll_deg.
        down_world = rot_c2w @ down
        g_k = down_world @ pose_k[:3, :3]
        roll = math.degrees(math.atan2(float(g_k[0]), float(g_k[1])))
        entry = {
            "frame": int(fi),
            "right_m": round(float(rel @ right), 2),
            "forward_m": round(float(rel @ fwd), 2),
            # true-vertical offset vs the reference camera (+ = higher up):
            # the one camera quantity the flattened map cannot show.
            # + 0.0 folds IEEE -0.0 into 0.0 so JSON never shows "-0.0"
            "up_m": round(float(-(rel @ down)), 2) + 0.0,
            "roll_deg": round(roll, 1) + 0.0,
            "heading_deg": None,
        }
        if norm > 0.15:  # keeps within ~8.6° of straight up/down arrow-less
            # + 0.0 folds IEEE -0.0 into 0.0 so JSON never shows "-0.0"
            entry["heading_deg"] = round(math.degrees(math.atan2(hu, hv)), 1) + 0.0
            cam_dirs.append((hu / norm, hv / norm))
        else:
            cam_dirs.append(None)
        cameras.append(entry)

    # Manual instances (scene configurations only): everything registered via
    # /add_instance, placed in the same ground frame so a mis-registered
    # marker is visible at a glance. position_w is stored unscaled; the
    # points/poses above were already multiplied by scene_scale, so the
    # marker must be too. Old configurations have no scene => no markers, no reply
    # key, byte-identical output.
    manual_markers: List[Dict[str, Any]] = []
    manual_draw: List[tuple] = []
    if scene is not None:
        for mname, rec in (scene.get("manual_instances") or {}).items():
            p_w = np.asarray(rec["position_w"], dtype=np.float64) * scene_scale
            rel = (p_w - cam_pos) @ rot_c2w
            mu, mv = float(rel @ right), float(rel @ fwd)
            heading = None
            dir_uv = None
            if rec.get("forward_w") is not None:
                d_ref = np.asarray(rec["forward_w"], dtype=np.float64) @ rot_c2w
                hu, hv = float(d_ref @ right), float(d_ref @ fwd)
                norm = math.hypot(hu, hv)
                if norm > 0.15:  # same near-vertical cutoff as camera wedges
                    heading = round(math.degrees(math.atan2(hu, hv)), 1) + 0.0
                    dir_uv = (hu / norm, hv / norm)
            manual_markers.append(
                {
                    "name": mname,
                    "right_m": round(mu, 2),
                    "forward_m": round(mv, 2),
                    "up_m": round(float(-(rel @ down)), 2) + 0.0,
                    "has_orientation": rec.get("forward_w") is not None,
                    "heading_deg": heading,
                }
            )
            manual_draw.append((mu, mv, mname, dir_uv))

    # --- compass: ground the anchors, pin north from the declared arrow ---
    compass_state: Optional[Dict[str, Any]] = None
    if compass_req is not None:
        # "scene centre" = the centre of the mapped footprint: midpoint of
        # the robust 1/99-percentile extent, measured over occupied ground
        # CELLS (one vote per 5 cm spot, like the object centroids) — a
        # room's centre, not anything density-weighted. Raw-point
        # percentiles were only mostly robust: a point pile at one end
        # outnumbers a sparse fringe and trims it out of the 1/99 window,
        # dragging the "centre" toward the pile.
        usable_u, usable_v = _ground_cells(u_all[keep], v_all[keep])
        scene_center = np.array(
            [
                float((np.percentile(usable_u, 1) + np.percentile(usable_u, 99)) / 2),
                float((np.percentile(usable_v, 1) + np.percentile(usable_v, 99)) / 2),
            ]
        )

        def _resolve_anchor(ref, raw_text, want_plane, side):
            kind, arg = ref
            if kind == "scene":
                return {
                    "kind": "point",
                    "p": scene_center.copy(),
                    "desc": "scene centre",
                    "scene": True,
                }
            if kind in ("camera", "camplane"):
                if arg not in cam_ground:
                    raise BadRequest(
                        f"compass.{side}: frame {arg} is not among the "
                        "reconstructed cameras",
                        camera_frames=poses_frames,
                    )
                if kind == "camera":
                    return {
                        "kind": "point",
                        "p": cam_ground[arg].copy(),
                        "desc": f"camera f{arg}",
                    }
                heading = cam_dir_of.get(arg)
                if heading is None:
                    raise BadRequest(
                        f"compass.{side}: camera f{arg} looks nearly "
                        "straight up/down — its image plane has no "
                        "direction on the ground plane"
                    )
                return {
                    "kind": "line",
                    "p": cam_ground[arg].copy(),
                    # the image plane's ground trace is perpendicular to
                    # the camera's ground heading
                    "d": np.array([heading[1], -heading[0]]),
                    "t_lo": -1.2,
                    "t_hi": 1.2,
                    "desc": f"camera f{arg} image plane",
                }
            fi, selector = arg
            matched = [
                c
                for c in candidates
                if c["frame"] == fi
                and (selector == str(c["id"]) or selector.lower() == c["label"].lower())
            ]
            if not matched:
                raise BadRequest(
                    f"compass.{side} {raw_text!r} matched no mapped instance "
                    f"in frame {fi} — erased or unsegmented objects cannot "
                    "anchor the compass",
                    available=sorted(
                        f"f{c['frame']}:{c['id']} {c['label']}"
                        for c in candidates
                        if c["frame"] == fi
                    ),
                )
            flat = np.concatenate([c["flat"] for c in matched])
            # footprint (occupied-cell) coordinates, not raw points: a
            # vertical face's stacked points must not drag the centroid or
            # the plane axis off the 2D footprint
            us, vs = _ground_cells(u_all[flat], v_all[flat])
            p = np.array([float(np.median(us)), float(np.median(vs))])
            desc = "+".join(sorted(f"f{c['frame']}:{c['id']}" for c in matched))
            desc += f" {matched[0]['label']}"
            if not want_plane:
                return {"kind": "point", "p": p, "desc": desc}
            # plane fit on the ground: the major PCA axis of the instance's
            # ground footprint (a wall-side object's trace along the
            # wall); its vertical plane is that line
            du, dv = us - us.mean(), vs - vs.mean()
            cov = np.array(
                [
                    [float(np.mean(du * du)), float(np.mean(du * dv))],
                    [float(np.mean(du * dv)), float(np.mean(dv * dv))],
                ]
            )
            evecs = np.linalg.eigh(cov)[1]
            axis = evecs[:, -1]
            t = (np.stack([us, vs], axis=1) - p) @ axis
            t_lo = float(np.percentile(t, 2.5))
            t_hi = float(np.percentile(t, 97.5))
            # A plane direction is only as real as the ground trace it is
            # fitted to: below half a metre the major axis is depth noise,
            # and the perpendicular foot slides metres along that arbitrary
            # line (v11/MMSI_11: a telephone's plane query read 63 deg off
            # its own centroid). Small objects are points, not planes.
            if t_hi - t_lo < 0.5:
                raise BadRequest(
                    f"compass.{side}: {desc} spans only {t_hi - t_lo:.2f} m "
                    "of ground — too small to define a plane direction; "
                    "small objects are points: use the centroid instead "
                    "(drop the plane flag)"
                )
            return {
                "kind": "line",
                "p": p,
                "d": axis,
                "t_lo": t_lo - 0.2,
                "t_hi": t_hi + 0.2,
                "desc": desc + " plane",
            }

        def _foot(point, line):
            rel_p = point - line["p"]
            return line["p"] + float(rel_p @ line["d"]) * line["d"]

        def _arrow_between(a, b, where):
            """Concrete arrow between two resolved anchors: (s, e, bearing)."""
            if a["kind"] == "line":
                e_pt = b["p"]
                s_pt = _foot(e_pt, a)
            elif b["kind"] == "line":
                s_pt = a["p"]
                e_pt = _foot(s_pt, b)
            else:
                s_pt, e_pt = a["p"], b["p"]
            vec = e_pt - s_pt
            if float(np.hypot(vec[0], vec[1])) < 0.05:
                raise BadRequest(
                    f"{where}: start and end land on the same map point "
                    "(< 0.05 m apart) — the direction is undefined; use "
                    "anchors separated on the ground",
                    start=a["desc"],
                    end=b["desc"],
                )
            return s_pt, e_pt, math.degrees(math.atan2(vec[0], vec[1])) % 360.0

        if c_orient is not None:
            start_anchor = _resolve_anchor(
                c_start_ref, str(compass_req.get("start")), c_start_plane, "start"
            )
            end_anchor = _resolve_anchor(
                c_end_ref, str(compass_req.get("end")), c_end_plane, "end"
            )
            s_pt, e_pt, anchor_bearing = _arrow_between(
                start_anchor, end_anchor, "compass anchor"
            )
            compass_state = {
                "start": start_anchor,
                "end": end_anchor,
                "s_pt": s_pt,
                "e_pt": e_pt,
                "orientation": c_orient,
                "anchor_bearing": anchor_bearing,
                "north_bearing": (anchor_bearing - _COMPASS_OFFSET_DEG[c_orient])
                % 360.0,
                "summary": (
                    f"{start_anchor['desc']} -> {end_anchor['desc']} is {c_orient}"
                ),
            }
            # Scene-conditional: the anchored north is also pinned in the
            # SCENE (world frame), so /compass queries and later maps on
            # ANY reference frame agree with this one.
            if scene is not None:
                _persist_scene_compass_from_bev(
                    work_base,
                    compass_state["north_bearing"],
                    rot_c2w,
                    right,
                    fwd,
                    compass_state["summary"],
                )
        else:
            # queries without a new anchor: the compass must already exist —
            # in the scene (world frame, any reference frame; new configurations) or
            # as this reference frame's saved bev_compass artifact.
            restored = _scene_compass_bearing(scene, rot_c2w, right, fwd)
            if restored is not None:
                compass_state = {
                    "restored": True,
                    "north_bearing": restored[0],
                    "summary": restored[1],
                }
            else:
                try:
                    prior = _newest_artifact(
                        "bev_compass", [frame], "compass", work_base
                    )
                except BadRequest:
                    prior = None
                if prior is None or prior.get("north_bearing_deg") is None:
                    raise BadRequest(
                        "compass queries need an anchored compass for this "
                        "reference frame — declare start/end/orientation (in "
                        "this call or an earlier one) first"
                    )
                prior_anchor = prior.get("anchor") or {}
                compass_state = {
                    "restored": True,
                    "north_bearing": float(prior["north_bearing_deg"]) % 360.0,
                    "summary": str(
                        prior_anchor.get("summary", "an earlier compass call")
                    ),
                }

        if c_queries:
            nb = compass_state["north_bearing"]

            def _diff(a, b):
                d = abs(a - b) % 360.0
                return min(d, 360.0 - d)

            query_results = []
            for k, q in enumerate(c_queries):
                qa = _resolve_anchor(
                    q["start_ref"],
                    q["start_raw"],
                    q["start_plane"],
                    f"queries[{k}].start",
                )
                qb = _resolve_anchor(
                    q["end_ref"], q["end_raw"], q["end_plane"], f"queries[{k}].end"
                )
                qs, qe, qbear = _arrow_between(qa, qb, f"compass.queries[{k}]")
                ranked = sorted(
                    _COMPASS_ORIENTATIONS,
                    key=lambda n: _diff(qbear, (nb + _COMPASS_OFFSET_DEG[n]) % 360.0),
                )
                nearest = [
                    {
                        "orientation": n,
                        "bearing_deg": round((nb + _COMPASS_OFFSET_DEG[n]) % 360.0, 1)
                        % 360.0,
                        "diff_deg": round(
                            _diff(qbear, (nb + _COMPASS_OFFSET_DEG[n]) % 360.0), 1
                        ),
                    }
                    # ALL EIGHT, ranked: a top-2 reply let the model argue
                    # itself into a direction it never saw the distance of
                    # (v10/104 answered one that was 121 deg off). With the
                    # full ranking, an absurd candidate is visibly absurd.
                    for n in ranked
                ]
                margin = round(nearest[1]["diff_deg"] - nearest[0]["diff_deg"], 1)
                ambiguous = margin < 10.0
                top, runner = nearest[0], nearest[1]
                # The conclusion sentence IS the tool's answer: naming the
                # nearest direction in prose leaves no gap for the model to
                # re-derive a different one from semantics or from squinting
                # at the map (v10/104 overrode a correct top-2; v11/104
                # overrode a correct all-8 ranking).
                # One sentence, whatever the margin. The reply used to
                # withhold the direction when the top two sat within 10 deg
                # ("too close to call - cross-check the images"), which is
                # the same override licence the teaching spent v15-v18
                # closing: a withheld answer is an invitation to invent one.
                # The closeness is reported as a FACT instead (margin_deg,
                # and the trailing clause below), and the ranking still
                # names its top entry.
                conclusion = (
                    f"{qa['desc']} -> {qb['desc']} points "
                    f"{top['orientation'].upper()} "
                    f"({top['diff_deg']:.1f} deg off; next candidate "
                    f"{runner['orientation']} is {runner['diff_deg']:.1f} "
                    "deg off)"
                )
                if ambiguous:
                    conclusion += (
                        f" — a close call: the top two are within "
                        f"{margin:.1f} deg of each other"
                    )
                # "Which side does view N see": the object -> camplane
                # arrow's compass direction IS the side seen (workflow rule
                # a). Stating that reading here closes the loophole where
                # the model re-derives the side from camera-position
                # semantics and overrides the readout (v11/MMSI_104).
                s_kind, e_kind = q["start_ref"][0], q["end_ref"][0]
                side_seen = cam_i = obj_desc = None
                if s_kind == "object" and e_kind == "camplane":
                    cam_i = q["end_ref"][1]
                    side_seen = top["orientation"]
                    obj_desc = qa["desc"]
                elif s_kind == "camplane" and e_kind == "object":
                    cam_i = q["start_ref"][1]
                    rbear = (qbear + 180.0) % 360.0
                    side_seen = min(
                        _COMPASS_ORIENTATIONS,
                        key=lambda n: _diff(
                            rbear, (nb + _COMPASS_OFFSET_DEG[n]) % 360.0
                        ),
                    )
                    obj_desc = qb["desc"]
                if side_seen is not None:
                    conclusion += (
                        f"; side-seen rule: view f{cam_i} sees the "
                        f"{obj_desc}'s {side_seen.upper()} side"
                    )
                query_results.append(
                    {
                        "s_pt": qs,
                        "e_pt": qe,
                        "start_raw": q["start_raw"],
                        "end_raw": q["end_raw"],
                        "start_desc": qa["desc"],
                        "end_desc": qb["desc"],
                        "bearing": qbear,
                        "direction": top["orientation"],
                        "conclusion": conclusion,
                        "nearest": nearest,
                        "margin": margin,
                        "ambiguous": ambiguous,
                        "scene": bool(qa.get("scene") or qb.get("scene")),
                    }
                )
            compass_state["queries"] = query_results
    else:
        # A compass anchored by an earlier call still applies to any later
        # map with the SAME reference frame (same ground frame — the
        # sidecar's frame_indices enforce the match), so split/erase maps
        # keep their orientation without re-anchoring. The scene's compass
        # (world frame; new configurations) is checked first — it survives
        # reference-frame changes; the artifact stays the old configurations' path.
        restored = _scene_compass_bearing(scene, rot_c2w, right, fwd)
        if restored is not None:
            compass_state = {
                "restored": True,
                "north_bearing": restored[0],
                "summary": restored[1],
            }
        else:
            try:
                prior = _newest_artifact("bev_compass", [frame], "compass", work_base)
            except BadRequest:
                prior = None
            if prior is not None and prior.get("north_bearing_deg") is not None:
                prior_anchor = prior.get("anchor") or {}
                compass_state = {
                    "restored": True,
                    "north_bearing": float(prior["north_bearing_deg"]) % 360.0,
                    "summary": str(
                        prior_anchor.get("summary", "an earlier compass call")
                    ),
                }

    # Map extent: robust percentiles over scene points plus every marker.
    bg_u, bg_v = u_all[keep], v_all[keep]
    lo_u, hi_u = np.percentile(bg_u, [1, 99])
    lo_v, hi_v = np.percentile(bg_v, [1, 99])
    for c in cameras:
        lo_u, hi_u = min(lo_u, c["right_m"]), max(hi_u, c["right_m"])
        lo_v, hi_v = min(lo_v, c["forward_m"]), max(hi_v, c["forward_m"])
    for mu, mv, _mname, _mdir in manual_draw:
        lo_u, hi_u = min(lo_u, mu), max(hi_u, mu)
        lo_v, hi_v = min(lo_v, mv), max(hi_v, mv)
    pad = 0.1 * max(hi_u - lo_u, hi_v - lo_v, 2.0)
    lo_u, hi_u, lo_v, hi_v = lo_u - pad, hi_u + pad, lo_v - pad, hi_v + pad
    if range_m:
        lo_u, hi_u = max(lo_u, -range_m), min(hi_u, range_m)
        lo_v, hi_v = max(lo_v, -range_m), min(hi_v, range_m)
    span = max(hi_u - lo_u, hi_v - lo_v, 2.0)

    size, margin = 900, 50
    scale = (size - 2 * margin) / span

    def to_px(u, v):
        return (
            int(margin + (u - lo_u) * scale),
            int(size - margin - (v - lo_v) * scale),
        )

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    # metre grid, drawn under everything
    grid_val = 235
    for gu in range(int(math.floor(lo_u)), int(math.ceil(hi_u)) + 1):
        x, _ = to_px(gu, 0)
        if 0 <= x < size:
            canvas[:, x] = grid_val
    for gv in range(int(math.floor(lo_v)), int(math.ceil(hi_v)) + 1):
        _, y = to_px(0, gv)
        if 0 <= y < size:
            canvas[y, :] = grid_val

    def scatter(us, vs, color):
        xs = (margin + (us - lo_u) * scale).astype(int)
        ys = (size - margin - (vs - lo_v) * scale).astype(int)
        ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        canvas[ys[ok], xs[ok]] = color

    step = max(1, bg_u.size // 150_000)
    scatter(bg_u[::step], bg_v[::step], np.array([205, 208, 206]))

    import colorsys

    color_of = {
        i: tuple(
            int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1.0, 0.85, 0.9)
        )
        for i in inst_points
    }
    for i, flat in inst_points.items():
        scatter(u_all[flat], v_all[flat], np.array(color_of[i]))

    img = Image.fromarray(canvas)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    font = _bev_font(16)
    small_font = _bev_font(13)

    for cam, direction in zip(cameras, cam_dirs):
        x, y = to_px(cam["right_m"], cam["forward_m"])
        ref = cam["frame"] == frame
        fill = (20, 60, 160) if ref else (100, 110, 130)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=fill)
        if direction is not None:
            # heading wedge along the camera's facing direction (for the
            # reference camera that is "up" the map by construction). Pixel y
            # grows downward, map v grows upward, hence the -dv.
            dx, dy = direction[0], -direction[1]
            tip = (x + 16 * dx, y + 16 * dy)
            base = (x + 4 * dx, y + 4 * dy)
            draw.polygon(
                [
                    tip,
                    (base[0] - 7 * dy, base[1] + 7 * dx),
                    (base[0] + 7 * dy, base[1] - 7 * dx),
                ],
                fill=fill,
            )
        draw.text((x + 8, y - 8), f"f{cam['frame']}", fill=fill, font=small_font)

    for k, obj in enumerate(objects):
        x, y = to_px(obj["right_m"], obj["forward_m"])
        color = color_of.get(k, (30, 30, 30))
        ax_info = axis_of.get(k)
        if ax_info is not None:
            # long-axis line under the dot: the drawn form of
            # footprint_axis_deg, so the angle the reply reports is the
            # angle the eye checks
            a = ax_info["p"] + ax_info["t_lo"] * ax_info["d"]
            b = ax_info["p"] + ax_info["t_hi"] * ax_info["d"]
            dark = tuple(int(c * 0.55) for c in color)
            draw.line([to_px(*a), to_px(*b)], fill=dark, width=3)
        # centroid dot first, label beside it: the dot marks the position the
        # bev_objects artifact stores, the text names it
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline=(30, 30, 30))
        text = f"{obj['shown_as']} {obj['label']}"
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + 7 + dx, y - 9 + dy), text, fill=(255, 255, 255), font=font)
        draw.text((x + 7, y - 9), text, fill=color, font=font)

    # add_instance markers: green diamonds (vs object dots / camera
    # wedges), label below-right to dodge the object labels; an arrow
    # shows the registered facing, same construction as camera wedges
    manual_color = (10, 125, 70)
    for mu, mv, mname, mdir in manual_draw:
        x, y = to_px(mu, mv)
        draw.polygon(
            [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)],
            fill=manual_color,
            outline=(30, 30, 30),
        )
        if mdir is not None:
            dx, dy = mdir[0], -mdir[1]
            tip = (x + 18 * dx, y + 18 * dy)
            base = (x + 6 * dx, y + 6 * dy)
            draw.polygon(
                [
                    tip,
                    (base[0] - 6 * dy, base[1] + 6 * dx),
                    (base[0] + 6 * dy, base[1] - 6 * dx),
                ],
                fill=manual_color,
            )
        for hdx, hdy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text(
                (x + 9 + hdx, y + 6 + hdy), mname, fill=(255, 255, 255), font=font
            )
        draw.text((x + 9, y + 6), mname, fill=manual_color, font=font)

    compass_color = (150, 30, 160)
    if compass_state is not None and not compass_state.get("restored"):
        for anchor in (compass_state["start"], compass_state["end"]):
            if anchor["kind"] == "line":
                a = anchor["p"] + anchor["t_lo"] * anchor["d"]
                b = anchor["p"] + anchor["t_hi"] * anchor["d"]
                _draw_dashed_line(draw, to_px(*a), to_px(*b), compass_color)
            if anchor.get("scene"):
                x, y = to_px(*anchor["p"])
                draw.line([(x - 7, y), (x + 7, y)], fill=compass_color, width=2)
                draw.line([(x, y - 7), (x, y + 7)], fill=compass_color, width=2)
                draw.text(
                    (x + 9, y + 4), "scene centre", fill=compass_color, font=small_font
                )
        s_px = to_px(*compass_state["s_pt"])
        e_px = to_px(*compass_state["e_pt"])
        _draw_arrow(draw, s_px, e_px, compass_color)
        draw.ellipse(
            [s_px[0] - 5, s_px[1] - 5, s_px[0] + 5, s_px[1] + 5],
            outline=compass_color,
            width=2,
        )
        label = _COMPASS_SHORT[compass_state["orientation"]]
        lx, ly = (s_px[0] + e_px[0]) / 2 + 8, (s_px[1] + e_px[1]) / 2 - 18
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((lx + dx, ly + dy), label, fill=(255, 255, 255), font=font)
        draw.text((lx, ly), label, fill=compass_color, font=font)
    if compass_state is not None:
        # query arrows: the read-out side of the compass, drawn in blue so
        # they never get mistaken for the declared (purple) anchor
        query_color = (30, 90, 190)
        for q in compass_state.get("queries") or []:
            qs_px, qe_px = to_px(*q["s_pt"]), to_px(*q["e_pt"])
            _draw_arrow(draw, qs_px, qe_px, query_color, width=3, head=11)
            top = q["nearest"][0]
            label = f"{_COMPASS_SHORT[top['orientation']]} {top['diff_deg']:.0f}"
            lx = (qs_px[0] + qe_px[0]) / 2 + 8
            ly = (qs_px[1] + qe_px[1]) / 2 + 4
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((lx + dx, ly + dy), label, fill=(255, 255, 255), font=font)
            draw.text((lx, ly), label, fill=query_color, font=font)
        _draw_compass_rose(
            draw,
            size - margin - 62,
            margin + 68,
            40,
            compass_state["north_bearing"],
            small_font,
            compass_color,
        )

    fused = ",".join(f"f{fi}" for fi in points_frames)
    draw.text(
        (margin, 8),
        f"BEV fused from {fused} — up = f{frame} forward (ground-projected), "
        f"wedge = camera facing, grid = 1 m",
        fill=(60, 66, 70),
        font=small_font,
    )
    header_y = 26
    if erased_resolved:
        # An erased map MUST say so on the image itself: two split maps
        # viewed side by side are indistinguishable without it. Own line so
        # a long erase list cannot push it off the right edge.
        draw.text(
            (margin, header_y),
            "ERASED: " + ", ".join(erased_resolved),
            fill=(180, 40, 30),
            font=small_font,
        )
        header_y += 18
    if compass_state is not None:
        # Same rule as ERASED: a compass-bearing map must say on the image
        # itself where north came from, or the rose looks like ground truth.
        prefix = (
            "COMPASS (from an earlier call)"
            if compass_state.get("restored")
            else "COMPASS"
        )
        draw.text(
            (margin, header_y),
            f"{prefix}: north = {compass_state['north_bearing']:.0f} deg "
            f"clockwise from map-up — anchor: {compass_state['summary']}",
            fill=compass_color,
            font=small_font,
        )

    # Split maps get a magnified inset of the remaining split-object
    # footprint: reading a handle direction off a 30-pixel blob is at the
    # edge of visual acuity (v12/376 read a +18 deg clockwise swing
    # backwards on the full-size map; the same swing is obvious at 4x).
    zoom_desc = None
    if erased_frame_labels:
        erased_labels = {lb for (_f, lb) in erased_frame_labels}
        targets = [c for c in candidates if c["label"].strip().lower() in erased_labels]
        if targets:
            tu = np.concatenate([u_all[c["flat"]] for c in targets])
            tv = np.concatenate([v_all[c["flat"]] for c in targets])
            x0, y0 = to_px(float(tu.min()), float(tv.max()))
            x1, y1 = to_px(float(tu.max()), float(tv.min()))
            pad = 14
            x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            x1, y1 = min(size, int(x1) + pad), min(size, int(y1) + pad)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                zoom = 4
                while zoom > 1 and max(x1 - x0, y1 - y0) * zoom > 320:
                    zoom -= 1
                inset = img.crop((x0, y0, x1, y1)).resize(
                    ((x1 - x0) * zoom, (y1 - y0) * zoom), Image.NEAREST
                )
                ix, iy = 6, size - inset.height - 6
                img.paste(inset, (ix, iy))
                draw.rectangle(
                    [ix - 1, iy - 1, ix + inset.width, iy + inset.height],
                    outline=(60, 66, 70),
                    width=2,
                )
                draw.rectangle([x0, y0, x1, y1], outline=(120, 126, 130), width=1)
                names = "+".join(sorted(f"f{c['frame']}:{c['id']}" for c in targets))
                label_txt = f"zoom x{zoom}: {names} {targets[0]['label']}"
                draw.text(
                    (ix + 4, max(2, iy - 18)),
                    label_txt,
                    fill=(60, 66, 70),
                    font=small_font,
                )
                zoom_desc = label_txt

    with artifact_writer(
        "bev", ".png", subdir="renders", work_base=work_base
    ) as writer:
        bev_path = writer.path
        with writer.open() as handle:
            img.save(handle, format="PNG")

    # The flattened 2D layout, saved for downstream numeric use: what the
    # rendered map shows as wedges and dots, as arrays. Headings are NaN for
    # a camera looking nearly straight up/down (same cases heading_deg is
    # null in the reply).
    cameras_arr = np.array(
        [
            [
                c["right_m"],
                c["forward_m"],
                c["up_m"],
                np.nan if c["heading_deg"] is None else c["heading_deg"],
            ]
            for c in cameras
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    cameras_path = save_array(
        cameras_arr,
        "bev_cameras",
        poses_frames,
        "bev_cameras",
        extra={
            "layout": (
                "(n_cameras, 4): right_m, forward_m, up_m, heading_deg per "
                "camera in the reference frame's gravity-aligned frame (up_m = "
                "true-vertical offset vs the reference camera); rows follow "
                "frame_indices; heading_deg is NaN when that camera looks "
                "nearly straight up/down"
            ),
            "reference_frame": frame,
            # Pixel <-> map geometry of THIS render, so later calls can
            # accept "point at the map" pixel coordinates (add_instance
            # --at-map, bev arrow endpoints) and convert them exactly.
            # Values are in the calibrated metres the map was drawn in;
            # map_scale is the calibration factor used, so raw world
            # units = metres / map_scale even if calibration changes
            # later.
            "render_px": {
                "size": size,
                "margin": margin,
                "scale": round(float(scale), 8),
                "lo_u": round(float(lo_u), 6),
                "lo_v": round(float(lo_v), 6),
                "map_scale": round(float(scene_scale), 8),
            },
        },
        work_base=work_base,
    )
    objects_arr = np.array(
        [[o["right_m"], o["forward_m"]] for o in objects], dtype=np.float64
    ).reshape(-1, 2)
    objects_path = save_array(
        objects_arr,
        "bev_objects",
        [frame],
        "bev_objects",
        extra={
            "layout": (
                "(n_objects, 2): ground-plane centroid right_m, forward_m per "
                "object, same frame as the map; rows align with 'labels'"
            ),
            "reference_frame": frame,
            "labels": [o["label"] for o in objects],
            "instances": [o["instances"] for o in objects],
        },
        work_base=work_base,
    )

    # The orientation→vector mapping, saved so the compass survives this
    # call: (8, 2) unit ground directions, one row per orientation. A later
    # /bev on the same reference frame rediscovers it and redraws the rose.
    compass_path = None
    compass_block = None
    if compass_state is not None:
        nb = compass_state["north_bearing"]
        directions = {
            name: round((nb + _COMPASS_OFFSET_DEG[name]) % 360.0, 1) % 360.0
            for name in _COMPASS_ORIENTATIONS
        }
        compass_block = {
            "north_bearing_deg": round(nb, 1) % 360.0,
            "directions_deg": directions,
            "note": (
                "bearings are clockwise from map-up, the same convention as "
                "heading_deg; the rose drawn top-right of the map shows the "
                "same directions. Compass relation from object P to object "
                "Q: bearing(P->Q) = atan2(Q.right_m - P.right_m, "
                "Q.forward_m - P.forward_m) in degrees — the nearest "
                "directions_deg entry is the answer"
            ),
        }
        if compass_state.get("queries"):
            compass_block["queries"] = [
                {
                    "start": q["start_raw"],
                    "end": q["end_raw"],
                    "start_grounded": q["start_desc"],
                    "end_grounded": q["end_desc"],
                    "bearing_deg": round(q["bearing"], 1) % 360.0,
                    "direction": q["direction"],
                    "conclusion": q["conclusion"],
                    "nearest": q["nearest"],
                    "margin_deg": q["margin"],
                    **(
                        {
                            "ambiguous_note": (
                                "the two nearest directions are within 10 deg "
                                "of each other (see margin_deg); reported so "
                                "the call can be described as close — the "
                                "ranking above still stands"
                            )
                        }
                        if q["ambiguous"]
                        else {}
                    ),
                }
                for q in compass_state["queries"]
            ]
            compass_block["queries_note"] = (
                "each query is the compass direction FROM its start TO its "
                "end (point-to-plane runs along the perpendicular), drawn as "
                "a blue arrow on the map; 'direction'/'conclusion' ARE the "
                "tool's answer — report that direction; 'nearest' ranks ALL "
                "EIGHT compass directions by "
                "angular distance as the supporting evidence — a direction "
                "far down the list cannot be the answer, no matter what the "
                "images suggest"
            )
        if compass_state.get("restored"):
            compass_block["restored_from_artifact"] = True
            compass_block["anchor"] = compass_state["summary"]
        else:
            s_pt, e_pt = compass_state["s_pt"], compass_state["e_pt"]
            anchor_info = {
                "start": str(compass_req.get("start")),
                "end": str(compass_req.get("end")),
                "start_grounded": compass_state["start"]["desc"],
                "end_grounded": compass_state["end"]["desc"],
                "orientation": compass_state["orientation"],
                "bearing_deg": round(compass_state["anchor_bearing"], 1) % 360.0,
                "start_xy_m": [round(float(s_pt[0]), 2), round(float(s_pt[1]), 2)],
                "end_xy_m": [round(float(e_pt[0]), 2), round(float(e_pt[1]), 2)],
                "summary": compass_state["summary"],
            }
            compass_block["anchor"] = anchor_info
            compass_arr = np.array(
                [
                    [
                        math.sin(math.radians((nb + _COMPASS_OFFSET_DEG[n]) % 360.0)),
                        math.cos(math.radians((nb + _COMPASS_OFFSET_DEG[n]) % 360.0)),
                    ]
                    for n in _COMPASS_ORIENTATIONS
                ],
                dtype=np.float64,
            )
            compass_path = save_array(
                compass_arr,
                "bev_compass",
                [frame],
                "bev_compass",
                extra={
                    "layout": (
                        "(8, 2): unit ground-plane direction (right_m, "
                        "forward_m) per compass orientation, rows follow "
                        "'orientations'; same axes and reference frame as "
                        "this map"
                    ),
                    "orientations": _COMPASS_ORIENTATIONS,
                    "reference_frame": frame,
                    "north_bearing_deg": round(nb, 1) % 360.0,
                    "anchor": anchor_info,
                },
                work_base=work_base,
            )

    content = {
        "ok": True,
        "frame": frame,
        "bev": to_caller_path(str(bev_path), alias, work_base=work_base),
        "extent_m": {
            "right": [round(float(lo_u), 1), round(float(hi_u), 1)],
            "forward": [round(float(lo_v), 1), round(float(hi_v), 1)],
        },
        "fused_frames": points_frames,
        "frames_with_masks": sorted(masks_envs),
        **(
            {
                "erased": erased_resolved,
                "erased_note": (
                    "these instances' points were removed from the fused "
                    "cloud before mapping; the map shows the object only "
                    "where the REMAINING frames saw it. For rotation, "
                    "read the remaining object's footprint_axis_deg here "
                    "and in the twin split map: wrap the difference "
                    "(later - earlier) into (-90, 90] — positive = "
                    "clockwise seen from above"
                ),
                **(
                    {
                        "footprint_zoom": (
                            zoom_desc + " — the bottom-left inset magnifies the "
                            "remaining split-object footprint; read the "
                            "handle/spur direction there, not off the "
                            "tiny full-map blob"
                        )
                    }
                    if zoom_desc
                    else {}
                ),
            }
            if erased_resolved
            else {}
        ),
        **(
            {
                "erase_warning": (
                    "still on this map: "
                    + ", ".join(erase_leftovers)
                    + " — the erase left other instance(s) with the SAME "
                    "label in the SAME frame, so the erased object may "
                    "still be drawn here. If the question means ONE such "
                    "object, view that frame's segment overlay, decide "
                    "which instance the question means (usually the one "
                    "being held or used), "
                    + "`drop` the others permanently, "
                    + "then re-run the split pair"
                ),
            }
            if erase_leftovers
            else {}
        ),
        "objects": objects,
        "objects_note": (
            "objects come from EVERY fused frame's masks — /segment is "
            "prompted, so these are the objects the question names; "
            "instances that share a label within "
            f"{merge_radius} m are merged into one entry (`instances` "
            "lists the fN:id of each view). Each is drawn as a dot at "
            "its ground-plane centroid; fN:id matches that frame's "
            "segment overlay. footprint_axis_deg is the ground bearing "
            "of the footprint's long axis (clockwise from map-up, mod "
            "180), drawn as a line through the dot; null for round or "
            "tiny footprints. For a split pair, wrap (axis_later - "
            "axis_earlier) into (-90, 90]: positive = the object "
            "turned CLOCKWISE seen from above"
        ),
        **({"unlabeled_instances": skipped} if skipped else {}),
        "cameras": cameras,
        "cameras_note": (
            "every camera is drawn as a dot with a wedge showing the "
            "direction it faces on the map; heading_deg is that facing, "
            "clockwise from map-up (0 = same way the reference camera "
            "faces, 90 = its right), null when the camera looks nearly "
            "straight up/down; up_m is the camera's true-vertical offset "
            "vs the reference camera — the map itself flattens height. "
            "roll_deg is the camera's in-image lean (+ = leans right): "
            "an angle read off two photos is only comparable after "
            "subtracting the two cameras' roll_deg — a roll change "
            "makes a still object LOOK rotated"
        ),
        **(
            {
                "manual_instances": manual_markers,
                "manual_instances_note": (
                    "green diamonds: instances registered via "
                    "add_instance, at their ground positions (arrow = "
                    "the direction the instance faces, projected onto "
                    "the map; heading_deg as for cameras; null when it "
                    "faces nearly straight up/down). Check the marker "
                    "sits where you meant — and its arrow points the "
                    "way you meant — before reading depth/side from it"
                ),
            }
            if manual_markers
            else {}
        ),
        **({"compass": compass_block} if compass_block else {}),
        "gravity_used": {
            "vertical_lean_deg": gravity_env.get("vertical_lean_deg"),
            "camera_pitch_deg": gravity_env.get("camera_pitch_deg"),
        },
        **(
            {
                "scale_calibration": {
                    "factor": round(scene_scale, 6),
                    "note": (
                        "every metre on this map is calibrated by the "
                        "scene's recorded scale factor (set via "
                        "/distance or /depth calibration)"
                    ),
                }
            }
            if scene is not None and scene_scale != 1.0
            else {}
        ),
        "axes_note": (
            "map coordinates are metres in the reference camera's ground "
            "frame: +right = its right, +forward = the direction it "
            "faces (projected onto the ground plane); the reference "
            "camera sits at (0, 0)"
        ),
        "artifacts": {
            "points": to_caller_path(
                points_env.get("resolved_path"), alias, work_base=work_base
            ),
            **(
                {
                    "masks": to_caller_path(
                        masks_envs[frame].get("resolved_path"),
                        alias,
                        work_base=work_base,
                    )
                }
                if frame in masks_envs
                else {}
            ),
            "gravity": to_caller_path(
                gravity_env.get("resolved_path"), alias, work_base=work_base
            ),
            "bev_cameras": to_caller_path(cameras_path, alias, work_base=work_base),
            "bev_objects": to_caller_path(objects_path, alias, work_base=work_base),
            **(
                {
                    "bev_compass": to_caller_path(
                        compass_path, alias, work_base=work_base
                    )
                }
                if compass_path
                else {}
            ),
        },
    }
    if scene is not None:
        content = _slim_scene_bev_reply(content)
    return JSONResponse(status_code=200, content=content)


def _slim_scene_bev_reply(content: Dict[str, Any]) -> Dict[str, Any]:
    """Scene configurations get the bev reply de-numbered: the map IMAGE carries the
    layout, and every quantitative fact has a conclusion-style endpoint
    (side/depth/distance/elevation/motion/compass) — raw coordinate and
    angle tables in the reply only invite in-head geometry, the exact
    behaviour the scene design exists to prevent. Old configurations (no scene)
    keep the full reply byte-for-byte."""
    slim = dict(content)
    for key in ("extent_m", "cameras", "cameras_note", "gravity_used", "artifacts"):
        slim.pop(key, None)
    slim["objects"] = [
        {k: row[k] for k in ("label", "shown_as", "instances", "n_views") if k in row}
        for row in content.get("objects") or []
    ]
    slim["objects_note"] = (
        "objects come from EVERY fused frame's masks — /segment is "
        "prompted, so these are the objects the question names; instances "
        "that share a label nearby are merged into one entry (`instances` "
        "lists the fN:id of each view). Each is drawn as a dot at its "
        "ground-plane centroid on the map; read their relations through "
        "the scene calls (side/distance/elevation/compass/motion), not by "
        "measuring the picture"
    )
    if "erased_note" in slim:
        slim["erased_note"] = (
            "these instances' points were removed from the fused cloud "
            "before mapping; the map shows the object only where the "
            "REMAINING frames saw it. The split-pair maps are the VISUAL "
            "check; the quantitative readout is `motion --of f<i>:X --of "
            "f<j>:X`, which states the shift and rotation outright"
        )
    if "manual_instances" in slim:
        slim["manual_instances"] = [
            {"name": row["name"], "has_orientation": row["has_orientation"]}
            for row in content["manual_instances"]
        ]
        slim["manual_instances_note"] = (
            "green diamonds: instances registered via add_instance, at "
            "their ground positions (arrow = the facing you registered, "
            "projected onto the map). Check the marker sits and points "
            "the way you meant before reading side from it"
        )
    if "compass" in slim and isinstance(slim["compass"], dict):
        kept = {
            k: slim["compass"][k]
            for k in ("anchor", "restored_from_artifact")
            if k in slim["compass"]
        }
        kept["note"] = (
            "the rose top-right of the map draws the anchored directions; "
            "read compass facts with compass queries — never by measuring "
            "angles off the map"
        )
        slim["compass"] = kept
    if "scale_calibration" in slim:
        slim["scale_calibration"] = {
            "note": (
                "every metre on this map is calibrated by the scene's "
                "recorded scale factor (set via /distance or /depth "
                "calibration)"
            )
        }
    return slim


# ---------------------------------------------------------------------------
# Persistent 3D scene: /recon boots it, /segment feeds it, /add_instance
# extends it, and /list_instance /distance /depth /compass read it back by
# instance NAME. State model in gateway/scene_store.py (disk snapshot is the
# source of truth; memory is a cache). Old configurations never call /recon, so no
# scene exists for them and every scene-conditional hook in the shared
# endpoints (/segment, /drop, /bev) is byte-for-byte inert.
# ---------------------------------------------------------------------------

#: How many frames GeoCalib runs on when the scene estimates gravity, and
#: how many of the best (lowest uncertainty) votes are fused.
SCENE_GRAVITY_MAX_FRAMES = 12
SCENE_GRAVITY_TOP_K = 3

#: Downsample caps: per-instance world-point sample and whole-cloud sample.
_SCENE_INST_MAX_PTS = 4000
_SCENE_CLOUD_MAX_PTS = 200_000

_SCENE_RESERVED_PREFIXES = ("camera:", "camplane:")

#: /add_instance orientation-by-axes: each pair names one body axis; the
#: caller picks WHICHEVER member of a pair has a nameable target in the
_AXIS_PAIRS = (("forward", "back"), ("right", "left"), ("top", "bottom"))


def _newest_bev_render(frame: int, work_base) -> Dict[str, Any]:
    """Newest bev_cameras sidecar whose REFERENCE frame is `frame`.

    _newest_artifact matches on the sidecar's frame list, but bev_cameras
    records every FUSED frame ([0,1,2]) — an exact [N] match never fires
    in a multi-frame scene (smoke14e MMSI_218: bev --frame 1 built, three
    --at-map f1 calls all bounced). The reference frame lives in the
    sidecar's own field, so filter on that."""
    base = Path(work_base) if work_base is not None else WORK_DIR
    best = None
    for sc in (base / "artifacts").glob("bev_cameras_*.npy.json"):
        try:
            meta = json.loads(sc.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("reference_frame") != frame:
            continue
        mtime = sc.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, meta)
    if best is None:
        raise BadRequest(f"no bev map for reference frame {frame}")
    return best[1]


def _bev_pixel_to_raw_uv(
    px: float, py: float, env: Dict[str, Any], where: str
) -> tuple[float, float]:
    """Invert one bev render's pixel mapping: PNG pixel -> RAW ground
    (u, v) in the reference camera's gravity-aligned frame. `env` is the
    newest bev_cameras envelope for that frame; its render_px sidecar
    block records the exact affine the map was drawn with."""
    r = env.get("render_px") or {}
    if not r:
        raise BadRequest(
            f"{where}: this frame's bev map predates pixel addressing — "
            "re-run bev for that frame and use the new map"
        )
    size, margin = int(r["size"]), int(r["margin"])
    if not (0 <= px < size and 0 <= py < size):
        raise BadRequest(
            f"{where}: pixel ({px:g}, {py:g}) is outside the map image ({size}x{size})",
        )
    u = (float(r["lo_u"]) + (px - margin) / float(r["scale"])) / float(r["map_scale"])
    v = (float(r["lo_v"]) + (size - margin - py) / float(r["scale"])) / float(
        r["map_scale"]
    )
    return u, v


def _scene_artifacts_dir(work_base: Optional[Path]) -> Path:
    return _validated_artifact_dir("artifacts", work_base or WORK_DIR)


def _scene_for(work_base: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The trial's scene, or None when /recon never ran (old configurations)."""
    base = work_base if work_base is not None else WORK_DIR
    artifacts = Path(base) / "artifacts"
    if not artifacts.is_dir():
        return None
    return scene_store.load(artifacts)


def _require_scene(work_base: Optional[Path]) -> Dict[str, Any]:
    scene = _scene_for(work_base)
    if scene is None or not scene.get("recon"):
        raise BadRequest(
            "no 3D scene exists for this trial yet — run /recon first (it "
            "reconstructs all frames and registers the cameras as instances)"
        )
    return scene


def _mutate_scene(work_base: Optional[Path], fn) -> Dict[str, Any]:
    """Serialize a scene mutation: lock, reload, apply, snapshot."""
    artifacts = _scene_artifacts_dir(work_base)
    with scene_store.lock_for(artifacts):
        scene = scene_store.load(artifacts)
        if scene is None:
            raise BadRequest("no 3D scene exists for this trial yet — run /recon first")
        scene = fn(scene)
        scene_store.save(artifacts, scene)
        return scene


def _scene_down(scene: Dict[str, Any]) -> Optional[np.ndarray]:
    grav = scene.get("gravity") or {}
    if grav.get("down_world") is None:
        return None
    return np.asarray(grav["down_world"], dtype=np.float64)


def _load_scene_points(scene: Dict[str, Any], work_base: Optional[Path]):
    recon = scene.get("recon") or {}
    path = recon.get("points_path")
    if not path:
        raise BadRequest("the scene has no reconstruction — run /recon first")
    try:
        return load_envelope(path, "points", work_base=work_base)
    except BadRequest as exc:
        raise ToolFailure(
            "the scene's reconstruction artifacts are no longer readable "
            "(cleaned up under the trial?) — re-run /recon",
            points_path=str(path),
        ) from exc


async def _ensure_scene_gravity(
    scene: Dict[str, Any], work_base: Optional[Path]
) -> Dict[str, Any]:
    """Scene gravity: GeoCalib over the frames, best-confidence votes fused.

    Runs at most once per scene (the per-frame estimates and the fused
    world down vector are persisted); concurrent callers race benignly —
    whoever saves first wins, the loser's identical work is discarded.
    """
    if _scene_down(scene) is not None:
        return scene
    frames_map = scene.get("frames") or {}
    poses_map = scene.get("camera_poses") or {}
    frame_ids = sorted(int(k) for k in frames_map)
    if not frame_ids:
        raise BadRequest("the scene has no frames — run /recon first")
    if len(frame_ids) > SCENE_GRAVITY_MAX_FRAMES:
        step = len(frame_ids) / SCENE_GRAVITY_MAX_FRAMES
        frame_ids = sorted(
            {frame_ids[int(i * step)] for i in range(SCENE_GRAVITY_MAX_FRAMES)}
        )

    per_frame: List[Dict[str, Any]] = []
    for fi in frame_ids:
        entry: Dict[str, Any] = {"frame": int(fi)}
        try:
            pose = np.asarray(poses_map[str(fi)], dtype=np.float64)
            image = load_image(Path(frames_map[str(fi)]))
            result = await call_upstream_async(
                "spatial_Gravity", "estimate", {"image_source": image}
            )
            raw = result if isinstance(result, dict) else vars(result)
            vector = as_numpy(raw.get("gravity_xyz"))
            if vector is None:
                raise ToolFailure("gravity estimator returned no gravity_xyz")
            vector = np.asarray(vector, dtype=np.float64).ravel()
            if (
                vector.shape != (3,)
                or not np.isfinite(vector).all()
                or np.linalg.norm(vector) < 1e-6
            ):
                raise ToolFailure(
                    "gravity estimator returned a degenerate vector",
                    got=vector.tolist(),
                )
            down_cam = vector / np.linalg.norm(vector)
            down_world = pose[:3, :3] @ down_cam
            uncertainty = raw.get("gravity_uncertainty_deg")
            entry.update(
                {
                    "down_cam": [round(float(v), 6) for v in down_cam],
                    "down_world": [round(float(v), 6) for v in down_world],
                    "uncertainty_deg": (
                        round(float(uncertainty), 2)
                        if isinstance(uncertainty, (int, float))
                        and math.isfinite(float(uncertainty))
                        else None
                    ),
                }
            )
        except (GatewayError, KeyError, OSError) as exc:
            entry["error"] = str(exc)
        per_frame.append(entry)

    try:
        fused = scene_store.aggregate_gravity(per_frame, top_k=SCENE_GRAVITY_TOP_K)
    except scene_store.SceneError as exc:
        raise ToolFailure(
            "scene gravity estimation failed on every sampled frame",
            per_frame=[
                {"frame": f["frame"], "error": f.get("error")} for f in per_frame
            ],
        ) from exc

    def _apply(current: Dict[str, Any]) -> Dict[str, Any]:
        if _scene_down(current) is None:
            current["gravity"] = {**fused, "per_frame": per_frame}
        return current

    return _mutate_scene(work_base, _apply)


def _unique_scene_name(base_name: str, taken: Dict[str, Any]) -> str:
    if base_name not in taken:
        return base_name
    n = 2
    while f"{base_name}_{n}" in taken:
        n += 1
    return f"{base_name}_{n}"


def _scene_registry(
    scene: Dict[str, Any],
    work_base: Optional[Path],
    with_points: bool = True,
):
    """Every live instance, derived fresh: (registry, ctx).

    registry: name -> record with position_w (raw world units) and, when
    oriented, the forward/right/up unit vectors. Segment-derived instances
    are recomputed from the NEWEST masks artifacts on every call, so a
    /drop or a re-/segment can never leave the scene stale (the same
    newest-wins contract as /bev's auto-discovery).

    ctx carries the shared geometry (points envelope, gravity basis, the
    ungrouped per-frame candidates for f<N>:<id> references).
    """
    registry: Dict[str, Dict[str, Any]] = {}
    down = _scene_down(scene)
    for fi_str in sorted(scene.get("camera_poses") or {}, key=int):
        pose = np.asarray(scene["camera_poses"][fi_str], dtype=np.float64)
        rot = pose[:3, :3]
        name = f"camera:f{int(fi_str)}"
        registry[name] = {
            "name": name,
            "kind": "camera",
            # The staged file's own name: `frame_<N>.png` for the still-image
            # benchmarks (byte-identical to the literal this used to be),
            # `v<K>_frame_<N>.png` under MMSI-Video.
            "source": "recon: camera pose of "
            + os.path.basename(
                str(
                    (scene.get("frames") or {}).get(fi_str)
                    or f"frame_{int(fi_str)}.png"
                )
            ),
            "position_w": pose[:3, 3].copy(),
            "forward_w": rot @ np.array([0.0, 0.0, 1.0]),
            "right_w": rot @ np.array([1.0, 0.0, 0.0]),
            "up_w": rot @ np.array([0.0, -1.0, 0.0]),
            "frame": int(fi_str),
        }

    ctx: Dict[str, Any] = {
        "down": down,
        "points_env": None,
        "candidates": [],
        "skipped": [],
    }

    if with_points and scene.get("recon"):
        points_env = _load_scene_points(scene, work_base)
        ctx["points_env"] = points_env
        points = np.asarray(points_env["array"], dtype=np.float64)
        points_frames = [int(i) for i in points_env["frame_indices"]]
        crop = points_env.get("crop_box") or {}
        left_off, top_off = int(crop.get("left", 0)), int(crop.get("top", 0))
        grid_h, grid_w = int(points.shape[1]), int(points.shape[2])
        dynamic_env = _optional_artifact(
            "dynamic", points_frames, "dynamic", work_base, contains=True
        )
        ctx["dynamic_env"] = dynamic_env

        # The scene centre is a GLOBAL query, so it reads static points
        # only — a mover must not drag the room footprint around.
        cloud = points.reshape(-1, 3)
        static_keep = _dynamic_flat_keep(
            dynamic_env, points_frames, grid_h, grid_w, anchor_frame=None
        )
        if static_keep is not None:
            cloud = cloud[static_keep]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if cloud.shape[0] > _SCENE_CLOUD_MAX_PTS:
            cloud = cloud[:: cloud.shape[0] // _SCENE_CLOUD_MAX_PTS + 1]
        if cloud.shape[0]:
            if down is not None:
                e1, e2 = scene_store.horizontal_basis(down)
                us, vs = cloud @ e1, cloud @ e2
                cu, cv = scene_store.ground_cells(us, vs)
                u_mid = float((np.percentile(cu, 1) + np.percentile(cu, 99)) / 2)
                v_mid = float((np.percentile(cv, 1) + np.percentile(cv, 99)) / 2)
                h_med = float(np.median(cloud @ down))
                centre = e1 * u_mid + e2 * v_mid + down * h_med
                centre_src = "recon: room footprint centre (gravity-aligned)"
            else:
                centre = np.median(cloud, axis=0)
                centre_src = (
                    "recon: point-cloud centre (refined to the footprint "
                    "centre once gravity is estimated)"
                )
            registry["scene"] = {
                "name": "scene",
                "kind": "scene",
                "source": centre_src,
                "position_w": centre,
                "forward_w": None,
                "right_w": None,
                "up_w": None,
            }

        dyn_arr = None
        dyn_frames: List[int] = []
        if dynamic_env is not None:
            arr = np.asarray(dynamic_env["array"])
            if arr.ndim == 3 and arr.shape[1:] == (grid_h, grid_w):
                dyn_arr = arr.astype(bool)
                dyn_frames = [int(i) for i in dynamic_env.get("frame_indices") or []]

        candidates: List[Dict[str, Any]] = []
        for lf, fi in enumerate(points_frames):
            try:
                env = _newest_artifact("masks", [fi], "masks", work_base)
            except BadRequest:
                continue
            frame_masks = np.asarray(env["array"])
            if frame_masks.ndim != 3:
                continue
            labels = list(env.get("labels") or [])
            dropped = {int(i) for i in env.get("dropped_ids") or []}
            coverage = frame_masks.reshape(frame_masks.shape[0], -1).mean(axis=1)
            dyn_frame = (
                dyn_arr[dyn_frames.index(int(fi))]
                if dyn_arr is not None and int(fi) in dyn_frames
                else None
            )
            for i in range(int(frame_masks.shape[0])):
                if i in dropped:
                    continue
                ys, xs = np.nonzero(frame_masks[i])
                xc, yc = xs - left_off, ys - top_off
                ok_px = (xc >= 0) & (xc < grid_w) & (yc >= 0) & (yc < grid_h)
                if not ok_px.any():
                    ctx["skipped"].append(f"f{fi}:{i}")
                    continue
                pts = points[lf, yc[ok_px], xc[ok_px]]
                pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.shape[0] > _SCENE_INST_MAX_PTS:
                    pts = pts[:: pts.shape[0] // _SCENE_INST_MAX_PTS + 1]
                if pts.shape[0] < 20:
                    ctx["skipped"].append(f"f{fi}:{i}")
                    continue
                dyn_frac = (
                    float(dyn_frame[yc[ok_px], xc[ok_px]].mean())
                    if dyn_frame is not None
                    else None
                )
                candidates.append(
                    {
                        "frame": int(fi),
                        "id": int(i),
                        "label": str(labels[i] if i < len(labels) else "region"),
                        "cov": float(coverage[i]),
                        "pts": pts,
                        "pos": scene_store.robust_position(pts, down),
                        "dyn_frac": dyn_frac,
                    }
                )
        ctx["candidates"] = candidates

        def _ground_dist(a: np.ndarray, b: np.ndarray) -> float:
            if down is None:
                return float(np.linalg.norm(a - b))
            return float(np.linalg.norm(scene_store.horizontal(a - b, down)))

        groups: List[Dict[str, Any]] = []
        for cand in sorted(candidates, key=lambda c: -c["cov"]):
            key = cand["label"].strip().lower()
            for grp in groups:
                if grp["key"] == key and (
                    _ground_dist(cand["pos"], grp["members"][0]["pos"])
                    <= scene_store.MERGE_RADIUS_M
                ):
                    grp["members"].append(cand)
                    break
            else:
                groups.append({"key": key, "label": cand["label"], "members": [cand]})

        for grp in groups:
            pts = np.concatenate([m["pts"] for m in grp["members"]])
            members = sorted(f"f{m['frame']}:{m['id']}" for m in grp["members"])
            name = _unique_scene_name(grp["label"], registry)
            fracs = [
                m["dyn_frac"] for m in grp["members"] if m.get("dyn_frac") is not None
            ]
            per_frame: Dict[str, List[np.ndarray]] = {}
            for m in grp["members"]:
                per_frame.setdefault(f"f{m['frame']}", []).append(m["pts"])
            registry[name] = {
                "name": name,
                "kind": "segment",
                "source": f"segment: {grp['label']} ({', '.join(members)})",
                "position_w": scene_store.robust_position(pts, down),
                "forward_w": None,
                "right_w": None,
                "up_w": None,
                "label": grp["label"],
                "members": members,
                "n_views": len({m["frame"] for m in grp["members"]}),
                "pts": pts,
                # Where each member frame saw it (the no-scene-memory configuration
                # hands these out so a mover can be referenced per moment).
                "views_w": {
                    f: scene_store.robust_position(np.concatenate(p), down)
                    for f, p in sorted(per_frame.items())
                },
                # Moving = the Pi3MOS dynamic mask covers a majority of the
                # instance's pixels in ANY member view. Absent (None) when
                # the scene has no dynamic artifact.
                **(
                    {
                        "moving": max(fracs) >= _DYNAMIC_INSTANCE_FRAC,
                        "moving_fraction": max(fracs),
                    }
                    if fracs
                    else {}
                ),
            }

    for name in sorted(scene.get("manual_instances") or {}):
        rec = scene["manual_instances"][name]
        entry = {
            "name": name,
            "kind": "manual",
            "source": str(rec.get("source", "add_instance")),
            "position_w": np.asarray(rec["position_w"], dtype=np.float64),
            "forward_w": None,
            "right_w": None,
            "up_w": None,
        }
        for axis in ("forward_w", "right_w", "up_w"):
            if rec.get(axis) is not None:
                entry[axis] = np.asarray(rec[axis], dtype=np.float64)
        if name in registry:
            # A manual instance predates a same-named derived one (added,
            # then the label was segmented). The manual one keeps the name
            # -- it was there first and /add_instance rejected collisions
            # at creation time -- and the derived one was already renamed
            # by _unique_scene_name only when IT came later. Renaming here
            # would break names the model has already used.
            pass
        registry[name] = entry
    return registry, ctx


def _resolve_scene_point(
    text: Any, registry: Dict[str, Dict[str, Any]], ctx: Dict[str, Any], field: str
) -> Dict[str, Any]:
    """A point-instance reference: registry name, f<N>:<label-or-id>, or a
    case-insensitive unique name match. Returns {name, desc, p3, record}."""
    raw = str(text or "").strip()
    if not raw:
        raise BadRequest(f"{field} must name an instance (see /list_instance)")
    rec = registry.get(raw)
    if rec is None:
        folded = [r for n, r in registry.items() if n.lower() == raw.lower()]
        if len(folded) == 1:
            rec = folded[0]
    if rec is None:
        head, sep, selector = raw.partition(":")
        head_l = head.strip().lower()
        if sep and head_l.startswith("f") and head_l.removeprefix("f").isdigit():
            fi = int(head_l.removeprefix("f"))
            selector = selector.strip()
            matched = [
                c
                for c in ctx.get("candidates") or []
                if c["frame"] == fi
                and (selector == str(c["id"]) or selector.lower() == c["label"].lower())
            ]
            if matched:
                pts = np.concatenate([c["pts"] for c in matched])
                pos = scene_store.robust_position(pts, ctx.get("down"))
                desc = "+".join(sorted(f"f{c['frame']}:{c['id']}" for c in matched))
                return {
                    "name": raw,
                    "desc": f"{desc} {matched[0]['label']}",
                    "p3": pos,
                    "record": {"kind": "segment", "pts": pts},
                }
    if rec is None:
        raise BadRequest(
            f"{field}: no instance named {raw!r} in the scene",
            available=sorted(registry),
        )
    return {
        "name": rec["name"],
        "desc": rec["name"],
        "p3": np.asarray(rec["position_w"], dtype=np.float64),
        "record": rec,
    }


def _scene_instance_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": rec["name"],
        "source": rec["source"],
        "has_orientation": rec.get("forward_w") is not None,
    }
    # Only present when the scene has Pi3MOS dynamic masks: whether
    # this object itself MOVES between frames (camera motion never
    # sets it). A moving object's cross-frame fusion is unreliable —
    # read its per-frame positions (f<N>:<label>) or /motion instead.
    if "moving" in rec:
        row["motion"] = "moving" if rec["moving"] else "static"
    return row


def _scene_listing(registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {"camera": 0, "scene": 1, "segment": 2, "manual": 3}
    rows = sorted(registry.values(), key=lambda r: (order.get(r["kind"], 9), r["name"]))
    return [_scene_instance_row(r) for r in rows]


def _scene_status(scene: Dict[str, Any]) -> Dict[str, Any]:
    factor = scene_store.scale_factor(scene)
    return {
        "frames": sorted(int(k) for k in scene.get("frames") or {}),
        "gravity_ready": _scene_down(scene) is not None,
        "compass_anchored": bool((scene.get("compass") or {}).get("north_world")),
        "scale_factor": round(factor, 6),
        "scale_calibrated": bool((scene.get("scale") or {}).get("history")),
    }


def _scene_units_note(scene: Dict[str, Any]) -> str:
    factor = scene_store.scale_factor(scene)
    if (scene.get("scale") or {}).get("history"):
        return (
            f"metres, calibrated against the question's stated measurement "
            f"(scale factor {factor:.4f} over the raw reconstruction)"
        )
    return "metres on the reconstruction's own metric scale (uncalibrated)"


def _segment_scene_extra(
    scene: Dict[str, Any], work_base: Optional[Path], prompt_list: List[str]
) -> Dict[str, Any]:
    """Additive /segment reply fields (only when a scene exists)."""
    try:
        registry, _ctx = _scene_registry(scene, work_base)
    except GatewayError as exc:
        return {
            "scene_note": (
                f"masks saved, but scene instance registration failed: {exc.message}"
            )
        }
    wanted = {p.strip().lower() for p in prompt_list}
    rows = [
        _scene_instance_row(r)
        for r in registry.values()
        if r["kind"] == "segment" and str(r.get("label", "")).strip().lower() in wanted
    ]
    return {
        "scene_instances": rows,
        "scene_note": (
            "each labeled object is now a named instance in the 3D scene "
            "(same-label detections within "
            f"{scene_store.MERGE_RADIUS_M} m merge into one). Read every "
            "spatial fact about them through the scene: distance, depth, "
            "compass — by these names; list_instance shows the registry"
        ),
    }


def _drop_scene_extra(
    scene: Dict[str, Any], work_base: Optional[Path]
) -> Dict[str, Any]:
    """Additive /drop reply fields (only when a scene exists)."""
    try:
        registry, _ctx = _scene_registry(scene, work_base)
    except GatewayError as exc:
        return {
            "scene_note": (
                f"drop recorded, but the scene registry could not be "
                f"refreshed: {exc.message}"
            )
        }
    return {
        "scene_instances": [
            _scene_instance_row(r) for r in registry.values() if r["kind"] == "segment"
        ],
        "scene_note": (
            "the scene registry derives from the newest masks, so these "
            "are the segment instances that remain after the drop"
        ),
    }


async def _scene_bev_gravity_env(
    scene: Dict[str, Any],
    work_base: Optional[Path],
    frame: int,
    poses_env: Dict[str, Any],
):
    """Gravity envelope for /bev from the SCENE (GeoCalib, best frames).

    Replaces the per-frame /gravity artifact requirement for scene configurations:
    the fused world down vector is rotated into the reference camera's
    coordinates and saved as a regular gravity artifact so the reply's
    artifact paths and any later legacy-style discovery behave as before.
    """
    scene = await _ensure_scene_gravity(scene, work_base)
    down_world = _scene_down(scene)
    poses_frames = [int(i) for i in poses_env["frame_indices"]]
    if frame not in poses_frames:
        raise BadRequest(
            "camera_poses does not cover the reference frame",
            poses_frames=poses_frames,
        )
    pose = np.asarray(poses_env["array"][poses_frames.index(frame)], dtype=np.float64)
    down_cam = pose[:3, :3].T @ down_world
    down_cam = down_cam / np.linalg.norm(down_cam)
    lean, pitch = _lean_pitch_from_gravity(down_cam)
    grav = scene.get("gravity") or {}
    path = save_array(
        down_cam,
        "gravity",
        [frame],
        "gravity",
        extra={
            "layout": (
                "(3,) unit gravity direction, camera coords x right / "
                "y down / z forward"
            ),
            "vertical_lean_deg": round(lean, 1),
            "camera_pitch_deg": round(pitch, 1),
            "method": str(grav.get("method", "scene geocalib")),
        },
        work_base=work_base,
    )
    return scene, {
        "array": down_cam,
        "frame_indices": [frame],
        "resolved_path": path,
        "kind": "gravity",
        "vertical_lean_deg": round(lean, 1),
        "camera_pitch_deg": round(pitch, 1),
    }


def _scene_compass_bearing(
    scene: Optional[Dict[str, Any]],
    rot_c2w: np.ndarray,
    right: np.ndarray,
    fwd: np.ndarray,
) -> Optional[tuple]:
    """(north_bearing_deg, summary) of the scene's stored compass in a BEV
    map's ground basis, or None when the scene has no compass."""
    stored = (scene or {}).get("compass") or {}
    if stored.get("north_world") is None:
        return None
    north_world = np.asarray(stored["north_world"], dtype=np.float64)
    north_ref = rot_c2w.T @ north_world
    nu, nv = float(north_ref @ right), float(north_ref @ fwd)
    if math.hypot(nu, nv) < 1e-6:
        return None
    return (
        math.degrees(math.atan2(nu, nv)) % 360.0,
        str(stored.get("summary", "the scene compass")),
    )


def _persist_scene_compass_from_bev(
    work_base: Optional[Path],
    north_bearing: float,
    rot_c2w: np.ndarray,
    right: np.ndarray,
    fwd: np.ndarray,
    summary: str,
) -> None:
    """A compass anchored through /bev also pins the SCENE's north, so
    /compass queries and later maps on any reference frame agree."""
    nb = math.radians(north_bearing)
    north_ref = math.sin(nb) * np.asarray(right) + math.cos(nb) * np.asarray(fwd)
    north_world = np.asarray(rot_c2w, dtype=np.float64) @ north_ref
    record = {
        "summary": summary,
        "via": "/bev",
        "at": datetime.now(timezone.utc).isoformat(),
    }

    def _apply(current: Dict[str, Any]) -> Dict[str, Any]:
        compass = current.get("compass") or {"history": []}
        if compass.get("north_world") is not None:
            record["replaced"] = compass.get("summary")
        compass["north_world"] = [round(float(v), 6) for v in north_world]
        compass["summary"] = summary
        compass.setdefault("history", []).append(record)
        current["compass"] = compass
        return current

    _mutate_scene(work_base, _apply)


def _apply_scene_calibration(
    work_base: Optional[Path],
    kind: str,
    a_desc: str,
    b_desc: str,
    stated: Any,
    raw_value: float,
) -> Dict[str, Any]:
    """Record a scale factor from a stated measurement: factor = stated/raw."""
    try:
        stated_f = float(stated)
    except (TypeError, ValueError) as exc:
        raise BadRequest("the stated measurement must be a number") from exc
    if not math.isfinite(stated_f) or stated_f <= 0:
        raise BadRequest("the stated measurement must be a positive number")
    if raw_value < 0.05:
        raise BadRequest(
            f"cannot calibrate on a raw {kind} of {raw_value:.3f} — the two "
            "anchors are too close (or the target is behind the observer); "
            "pick a pair with real separation"
        )
    factor = stated_f / raw_value
    record = {
        "kind": kind,
        "from": a_desc,
        "to": b_desc,
        "stated": stated_f,
        "raw": round(raw_value, 4),
        "factor": round(factor, 6),
        "at": datetime.now(timezone.utc).isoformat(),
    }

    def _apply(current: Dict[str, Any]) -> Dict[str, Any]:
        scale = current.setdefault("scale", {"factor": 1.0, "history": []})
        record["previous_factor"] = round(float(scale.get("factor", 1.0)), 6)
        scale["factor"] = factor
        scale.setdefault("history", []).append(record)
        return current

    _mutate_scene(work_base, _apply)
    reply = {
        "calibrated": True,
        "scale_factor": round(factor, 6),
        "based_on": f"{kind} {a_desc} -> {b_desc}: stated {stated_f} over raw {raw_value:.3f}",
    }
    if abs(record["previous_factor"] - 1.0) > 1e-9:
        reply["replaced_factor"] = record["previous_factor"]
    return reply


#: Adjacent frames whose mutual covisibility (both directions) falls below
#: this fraction get named in /recon's overlap_warning. Provisional — set
#: from the MMSI_6 family's shape (a through-door pair shares almost
#: nothing), to be calibrated against real trials.
_OVERLAP_WARN_FRAC = 0.05


def _covis_frac(
    pts_a: np.ndarray, depth_b: np.ndarray, pose_b: np.ndarray, k_b: np.ndarray
) -> float:
    """Fraction of frame a's points that frame b plausibly also sees.

    A point counts when it lands inside b's frustum AND agrees with b's own
    reconstructed depth at that pixel. The depth gate is what gives the
    SELF-REPORTED overlap some teeth against a wrong registration: two
    views of different rooms forced into nearby poses still project into
    each other's frustum, but their depths clash.
    """
    rot, t = pose_b[:3, :3], pose_b[:3, 3]
    cam = (pts_a - t) @ rot  # world -> b's camera coords (R^T (X - t))
    z = cam[:, 2]
    front = z > 0.05
    if not front.any():
        return 0.0
    cam, z = cam[front], z[front]
    u = cam[:, 0] / z * float(k_b[0, 0]) + float(k_b[0, 2])
    v = cam[:, 1] / z * float(k_b[1, 1]) + float(k_b[1, 2])
    h, w = depth_b.shape
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inb.any():
        return 0.0
    d = depth_b[v[inb].astype(int), u[inb].astype(int)]
    z_in = z[inb]
    good = np.isfinite(d) & (np.abs(z_in - d) <= np.maximum(0.3, 0.15 * d))
    return float(good.sum()) / float(pts_a.shape[0])


def _recon_overlap(extras: Dict[str, Any]):
    """(warning text or None, low-overlap pairs) — TWO-VIEW scenes only.

    Restricted to exactly two frames (user decision): that is where the
    through-door failure lives and where one seam is actionable — with
    more views a jump is usually bridged by the frames between, and an
    adjacent-pair chain over a long sequence mostly yields noise.

    Self-reported covisibility (the user-picked trade-off): free — no
    extra GPU calls — but computed inside the reconstruction's own
    geometry, so a wrong registration that happens to be self-consistent
    can evade it; the depth gate in _covis_frac is the counterweight.
    Skipped entirely (no warning either way) when the upstream returned
    no intrinsics.
    """
    points = extras.get("points")
    depth = extras.get("depth")
    intrinsics = extras.get("intrinsics")
    poses = extras.get("poses")
    if points is None or depth is None or intrinsics is None or poses is None:
        return None, []
    frames = [int(i) for i in extras["frame_indices"]]
    if len(frames) != 2:
        return None, []
    order = sorted(range(len(frames)), key=lambda i: frames[i])

    def _sample(i: int) -> np.ndarray:
        p = points[i].reshape(-1, 3).astype(np.float64)
        p = p[np.isfinite(p).all(axis=1)]
        if p.shape[0] > 8192:
            p = p[:: p.shape[0] // 8192 + 1]
        return p

    low: List[List[int]] = []
    for k in range(len(order) - 1):
        ia, ib = order[k], order[k + 1]
        pa, pb = _sample(ia), _sample(ib)
        if pa.shape[0] < 100 or pb.shape[0] < 100:
            continue
        f_ab = _covis_frac(pa, depth[ib], poses[ib], intrinsics[ib])
        f_ba = _covis_frac(pb, depth[ia], poses[ia], intrinsics[ia])
        # max: a zoom-in pair (one view a close-up inside the other) is
        # healthy covisibility; warn only when NEITHER side sees the other
        if max(f_ab, f_ba) < _OVERLAP_WARN_FRAC:
            low.append([frames[ia], frames[ib]])
    if not low:
        return None, []
    pair_txt = ", ".join(f"f{a}-f{b}" for a, b in low)
    note = (
        f"WARNING: the two views {pair_txt} share almost no covisible "
        "view — the camera may have jumped (through a doorway, or a cut), "
        "and the reconstruction stitches those moments together on THIN "
        "evidence. Readouts that SPAN such a seam (side/distance/"
        "elevation/motion between things seen on opposite sides of it, "
        "and the fused map across it) can be silently wrong: prefer "
        "relations among things seen within the same span, and "
        "double-check anything that crosses the seam against the "
        "per-frame views."
    )
    return note, low


async def _recon_dynamic(
    extras: Dict[str, Any], work_base: Optional[Path]
) -> Optional[Dict[str, str]]:
    """Pi3MOS side-channel: per-pixel motion probability + hysteresis mask.

    Saved as two artifacts on the SAME (cropped) grid as the points
    artifact, so flat frame-major indices line up: `dynamic` (uint8, the
    core+flood rule) and `dynamic_prob` (float16, for threshold retuning
    without a rerun). The /recon reply deliberately says nothing about
    them — the signal surfaces as /segment's per-instance motion tag and
    as the static-cloud rule inside /bev, at_map and the scene centre.

    Any failure degrades to "no dynamic artifacts" (logged): a motion
    side-channel must never fail a reconstruction, and every consumer
    treats the artifacts as optional.
    """
    try:
        images = [load_image(p) for p in extras["image_paths"]]
        frame_indices = [int(i) for i in extras["frame_indices"]]
        result = await call_upstream_async(
            "spatial_Dynamic",
            "dynamic_masks",
            {"video_frames": images, "frame_indices": frame_indices},
        )
        prob = as_numpy(getattr(result, "dynamic_prob", None))
        if prob is None or prob.ndim != 3:
            raise ToolFailure("dynamic_masks returned no probability map")

        # Hysteresis on the model grid (cheap), then both arrays onto the
        # full original grid, then DA3's crop — after which they share the
        # points artifact's pixel grid exactly.
        binary = np.stack([_hysteresis_dynamic(p) for p in prob.astype(np.float32)])
        full_h, full_w = images[0].height, images[0].width
        prob_full = resize_spatial(prob.astype(np.float32), full_h, full_w)
        binary_full = resize_spatial(binary.astype(np.uint8), full_h, full_w)
        crop = extras.get("crop_box") or {}
        if crop:
            top, left = int(crop["top"]), int(crop["left"])
            ch, cw = int(crop["height"]), int(crop["width"])
            prob_full = prob_full[:, top : top + ch, left : left + cw]
            binary_full = binary_full[:, top : top + ch, left : left + cw]

        extra = {
            "crop_box": extras.get("crop_box"),
            "rule": {
                "core": _DYNAMIC_CORE,
                "floor": _DYNAMIC_FLOOR,
                "note": (
                    "dynamic = pixels >= floor 4-connected to a >= core "
                    "pixel (Pi3MOS motion probability)"
                ),
            },
        }
        bin_path = save_array(
            binary_full.astype(np.uint8),
            "dynamic",
            frame_indices,
            "dynamic",
            extra=extra,
            work_base=work_base,
        )
        prob_path = save_array(
            prob_full.astype(np.float16),
            "dynamic_prob",
            frame_indices,
            "dynamic_prob",
            extra=extra,
            work_base=work_base,
        )
        return {"dynamic_path": bin_path, "dynamic_prob_path": prob_path}
    except Exception as exc:  # noqa: BLE001 - side-channel must never fail /recon
        log_event(
            {
                "event": "recon_dynamic_degraded",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None


@app.post("/recon")
async def recon(request: Request) -> JSONResponse:
    """/reconstruct plus a persistent scene: cameras become instances.

    Body is the same as /reconstruct. On top of the reconstruction reply,
    the scene for this trial is (re)initialized: every camera pose is
    registered as instance "camera:f<N>" (position AND orientation), the
    room centre as "scene", and any already-saved segment masks become
    named instances immediately. A re-run resets the scene: a new
    reconstruction is a new world coordinate system, so instances,
    gravity, compass and calibration from the old one would silently be
    wrong in it.
    """
    payload = await read_json_body(request)
    body, extras = await _reconstruct_core(payload)
    work_base = extras["work_base"]
    # Scene configurations read the world through instance-name endpoints, so the
    # raw reconstruction numbers (4x4 poses, intrinsics, depth stats,
    # array paths) are withheld: every one of them is an invitation to
    # do geometry in-head, which is exactly what the scene design exists
    # to prevent. /reconstruct (the old configurations' endpoint) keeps them.
    body = {k: body[k] for k in ("found", "frame_indices", "num_frames") if k in body}

    scene = scene_store.new_scene()
    scene["frames"] = {
        str(fi): path
        for fi, path in zip(extras["frame_indices"], extras["image_paths"])
    }
    dyn_paths = await _recon_dynamic(extras, work_base)
    overlap_note, low_overlap_pairs = _recon_overlap(extras)
    scene["recon"] = {
        "points_path": extras["host_arrays"].get("points"),
        "depth_path": extras["host_arrays"].get("depth"),
        "camera_poses_path": extras["host_arrays"].get("camera_poses"),
        "frame_indices": [int(i) for i in extras["frame_indices"]],
        "crop_box": extras["crop_box"],
        "image_size": extras["image_size"],
        "metric_scale": extras["metric_scale"],
        "low_overlap_pairs": low_overlap_pairs,
        **(dyn_paths or {}),
    }
    scene["camera_poses"] = {
        str(fi): np.asarray(extras["poses"][i], dtype=np.float64).round(9).tolist()
        for i, fi in enumerate(extras["frame_indices"])
    }

    artifacts = _scene_artifacts_dir(work_base)
    with scene_store.lock_for(artifacts):
        had_scene = scene_store.load(artifacts) is not None
        scene_store.save(artifacts, scene)

    registry, _ctx = _scene_registry(scene, work_base)
    body["scene"] = {
        "initialized": True,
        **(
            {
                "reset_note": (
                    "an earlier scene for this trial was replaced — a new "
                    "reconstruction is a new coordinate system, so its instances, "
                    "compass and calibration no longer applied"
                )
            }
            if had_scene
            else {}
        ),
        **({"overlap_warning": overlap_note} if overlap_note else {}),
        "instances": _scene_listing(registry),
        "note": (
            "the 3D scene is live and persistent: /segment now registers "
            "each named object as an instance automatically; add_instance "
            "adds your own points; distance / depth / compass read "
            "measurements between instances BY NAME; list_instance shows "
            "what exists. All spatial readouts should come from these "
            "scene calls, not from eyeballing the images"
        ),
    }
    return JSONResponse(status_code=200, content=body)


@app.post("/list_instance")
async def list_instance(request: Request) -> JSONResponse:
    """Every instance in the scene: name, origin, orientation flag."""
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    registry, ctx = _scene_registry(scene, work_base)
    rows = _scene_listing(registry)
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "n": len(rows),
            "instances": rows,
            "scene_status": _scene_status(scene),
            **(
                {"unpositioned": sorted(set(ctx["skipped"]))}
                if ctx.get("skipped")
                else {}
            ),
            "note": (
                "these names are the handles every scene readout takes: "
                "distance --a/--b, depth --of/--observer, compass "
                "--start/--end/--query. Segment-derived instances track "
                "the newest masks — a drop or re-segment updates them"
            ),
        },
    )


async def _scene_up_hint(
    scene: Dict[str, Any], work_base: Path
) -> tuple[Dict[str, Any], np.ndarray, str]:
    """World-up for building an orientation frame: scene gravity, falling
    back to the mean camera up axis when gravity estimation fails."""
    try:
        scene = await _ensure_scene_gravity(scene, work_base)
        return scene, -_scene_down(scene), "scene gravity"
    except ToolFailure:
        ups = [
            np.asarray(p, dtype=np.float64)[:3, :3] @ np.array([0.0, -1.0, 0.0])
            for p in (scene.get("camera_poses") or {}).values()
        ]
        up = scene_store.unit(np.mean(np.stack(ups), axis=0))
        return scene, up, "mean camera up (gravity estimation failed)"


_PLAIN_AXIS_KEYS = tuple(k for pair in _AXIS_PAIRS for k in pair)

_ORIENTATION_GRAMMAR = (
    'orientation forms: {"facing": value} — standing upright, facing that '
    'way ({"facing_camera": N} is its camera shorthand); one bare '
    'axis, e.g. {"forward": value} — upright, that body axis turned that '
    'way; {"primary": {axis: value}, "secondary": {axis: value}} — '
    "the PRIMARY axis is taken exactly, the SECONDARY is projected onto "
    "the plane orthogonal to it (axis: forward/back/right/left/top/bottom); "
    'values are an instance name, the literal "gravity" (the downward '
    'direction), or "bev:f<N>:START:END" — an arrow read off the bev '
    "map rendered from frame N, each endpoint a PX,PY pixel on that "
    "map's PNG or a colon-free instance name; or legacy three bare "
    "axes, one per pair"
)

_BEV_VECTOR_USAGE = (
    'a bev map arrow is "bev:f<N>:START:END" — N the frame the map was '
    "rendered from, then the arrow's start and end. Each endpoint is "
    'a "PX,PY" pixel ON that frame\'s newest bev map PNG (point at the '
    "map the way you point at any image) or a colon-free instance name "
    '(e.g. "bev:f0:bed:door" points the way bed -> door lies on the '
    "map); the direction is start -> end. To face a camera, give "
    "camera:f<N> directly as the value instead of an arrow"
)


def _is_bev_vector(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith("bev:")


async def _scene_orientation(
    orientation: Any,
    pos_w: np.ndarray,
    registry: Dict[str, Any],
    ctx: Dict[str, Any],
    scene: Dict[str, Any],
    work_base: Path,
) -> tuple[Dict[str, Any], Dict[str, List[float]], str, Any]:
    """Build a new instance's (forward, right, up) axes from the orientation
    payload. Every frame is a PRIMARY axis (exact) + a SECONDARY direction
    (projected); facing and bare-axis forms are upright sugar (primary
    bottom = gravity). Returns (scene, axes, note, provenance)."""
    if not isinstance(orientation, dict) or not orientation:
        raise BadRequest(_ORIENTATION_GRAMMAR)
    facing_keys = [
        k for k in ("facing", "facing_camera") if orientation.get(k) is not None
    ]
    plain = [k for k in _PLAIN_AXIS_KEYS if orientation.get(k)]
    ps_keys = [k for k in ("primary", "secondary") if orientation.get(k) is not None]
    unknown = (
        set(orientation)
        - set(_PLAIN_AXIS_KEYS)
        - {"facing", "facing_camera", "primary", "secondary"}
    )
    if unknown:
        raise BadRequest("unknown orientation keys", got=sorted(unknown))
    if sum(1 for group in (facing_keys, plain, ps_keys) if group) > 1:
        raise BadRequest(
            "give ONE orientation form, not a mix. " + _ORIENTATION_GRAMMAR,
            got=sorted(facing_keys + plain + ps_keys),
        )

    async def _gravity_dir() -> tuple[np.ndarray, str]:
        nonlocal scene
        scene, up_hint, up_src = await _scene_up_hint(scene, work_base)
        return -up_hint, up_src

    def _target_dir(value: Any, label: str) -> tuple[np.ndarray, Dict[str, Any]]:
        ref = _resolve_scene_point(value, registry, ctx, label)
        d = np.asarray(ref["p3"], dtype=np.float64) - pos_w
        if float(np.linalg.norm(d)) < 0.05:
            raise BadRequest(
                f"{label}: {ref['desc']} sits on the new instance's own "
                "position — it cannot define a direction"
            )
        return d, ref

    async def _bev_vector_dir(value: str, label: str) -> tuple[np.ndarray, str]:
        """A direction read off the bev map as a start -> end arrow: each
        endpoint is a "PX,PY" pixel on that frame's newest bev map PNG,
        or a colon-free instance name. Same ground basis as the /bev
        render (the reference camera's ground-projected forward/right,
        image-up fallback for a near-vertical camera), rotated to world."""
        parts = value.split(":")
        head = parts[1].strip().lower().removeprefix("f") if len(parts) == 4 else ""
        if len(parts) != 4 or not head.isdigit():
            raise BadRequest(f"{label}: {_BEV_VECTOR_USAGE}", got=value)
        cam = registry.get(f"camera:f{int(head)}")
        if cam is None:
            raise BadRequest(
                f"{label}: frame {int(head)} is not among the reconstructed cameras",
                got=value,
            )
        down, _src = await _gravity_dir()
        factor = scene_store.scale_factor(scene)
        fwd = np.asarray(cam["forward_w"], dtype=np.float64)
        fwd = fwd - down * float(fwd @ down)
        if float(np.linalg.norm(fwd)) < 0.2:
            fwd = np.asarray(cam["up_w"], dtype=np.float64)
            fwd = fwd - down * float(fwd @ down)
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(down, fwd)
        right = right / np.linalg.norm(right)
        cam_p = np.asarray(cam["position_w"], dtype=np.float64)
        bev_env_cache: List[Any] = []

        def _bev_env() -> Dict[str, Any]:
            if not bev_env_cache:
                try:
                    bev_env_cache.append(_newest_bev_render(int(head), work_base))
                except BadRequest:
                    raise BadRequest(
                        f"{label}: pixel endpoints need the frame-{int(head)} "
                        f"bev map — run bev --frame {int(head)} and LOOK at "
                        "it first (instance-name endpoints work without it)"
                    ) from None
            return bev_env_cache[0]

        def _endpoint(token: str, which: str) -> tuple[np.ndarray, str]:
            token = token.strip()
            try:
                x, y = (float(v) for v in token.split(","))
            except ValueError:
                ref = _resolve_scene_point(
                    token, registry, ctx, f"{label} ({which} endpoint)"
                )
                return np.asarray(ref["p3"], dtype=np.float64), ref["desc"]
            u, v = _bev_pixel_to_raw_uv(x, y, _bev_env(), f"{label} ({which} endpoint)")
            return cam_p + u * right + v * fwd, f"(px {x:g}, {y:g})"

        a_pt, a_desc = _endpoint(parts[2], "start")
        b_pt, b_desc = _endpoint(parts[3], "end")
        d = b_pt - a_pt
        d = d - down * float(d @ down)
        if float(np.linalg.norm(d)) * factor < 0.05:
            raise BadRequest(
                f"{label}: the arrow's start and end land on the same map "
                "point (< 0.05 m apart on the ground) — the direction is "
                "undefined",
                got=value,
            )
        return d, f"bev:f{int(head)} map arrow {a_desc} -> {b_desc}"

    if facing_keys:
        if len(facing_keys) > 1:
            raise BadRequest("give facing OR facing_camera, not both")
        facing_key = facing_keys[0]
        if facing_key == "facing_camera":
            try:
                cam_frame = int(orientation["facing_camera"])
            except (TypeError, ValueError) as exc:
                raise BadRequest(
                    "orientation.facing_camera must be a frame index"
                ) from exc
            if registry.get(f"camera:f{cam_frame}") is None:
                raise BadRequest(
                    f"orientation.facing_camera: frame {cam_frame} is not "
                    "among the reconstructed cameras"
                )
            target = f"camera:f{cam_frame}"
        else:
            target = str(orientation["facing"])
        if _is_bev_vector(target):
            toward, facing_desc = await _bev_vector_dir(
                target.strip(), f"orientation.{facing_key}"
            )
            prov: Any = {"facing": target.strip()}
        else:
            toward, ref = _target_dir(target, f"orientation.{facing_key}")
            facing_desc = ref["desc"]
            prov = (
                {"facing_camera": cam_frame}
                if facing_key == "facing_camera"
                else {"facing": ref["name"]}
            )
        down, up_src = await _gravity_dir()
        f, r, u = scene_store.frame_from_primary_secondary(
            "bottom", down, "forward", toward
        )
        note = f"upright ({up_src}), facing {facing_desc}"
    elif ps_keys:

        def _axis_value(part: Any, label: str) -> tuple[str, str]:
            if not isinstance(part, dict) or len(part) != 1:
                raise BadRequest(
                    f"{label} must be a one-entry mapping {{axis: value}} — "
                    "axis one of forward/back/right/left/top/bottom, value "
                    'an instance name, "gravity", or "bev:f<N>:START:END"'
                )
            ((axis, value),) = part.items()
            if axis not in _PLAIN_AXIS_KEYS:
                raise BadRequest(
                    f"{label}: unknown axis {axis!r}",
                    expected=list(_PLAIN_AXIS_KEYS),
                )
            if not isinstance(value, str) or not value.strip():
                raise BadRequest(
                    f"{label}.{axis}: value must be an instance name, "
                    '"gravity", or "bev:f<N>:START:END"'
                )
            return axis, value.strip()

        if orientation.get("primary") is None:
            raise BadRequest("secondary given without primary. " + _ORIENTATION_GRAMMAR)
        p_axis, p_val = _axis_value(orientation["primary"], "orientation.primary")
        if orientation.get("secondary") is None:
            if p_axis in ("top", "bottom") or p_val == "gravity":
                raise BadRequest(
                    "this primary already pins the vertical, so the default "
                    "secondary (bottom = gravity) adds nothing — give an "
                    "explicit secondary to fix the heading, e.g. "
                    '{"secondary": {"top": <head-end instance>}} for someone '
                    "lying down"
                )
            s_axis, s_val, s_defaulted = "bottom", "gravity", True
        else:
            s_axis, s_val = _axis_value(
                orientation["secondary"], "orientation.secondary"
            )
            s_defaulted = False

        async def _dir_and_desc(
            axis: str, value: str, label: str
        ) -> tuple[np.ndarray, str, str]:
            if value == "gravity":
                down, up_src = await _gravity_dir()
                return down, f"gravity ({up_src})", "gravity"
            if _is_bev_vector(value):
                d, desc = await _bev_vector_dir(value, f"{label}.{axis}")
                return d, f"along {desc}", value
            d, ref = _target_dir(value, f"{label}.{axis}")
            return d, f"toward {ref['desc']}", ref["name"]

        p_dir, p_desc, p_name = await _dir_and_desc(
            p_axis, p_val, "orientation.primary"
        )
        s_dir, s_desc, s_name = await _dir_and_desc(
            s_axis, s_val, "orientation.secondary"
        )
        f, r, u = scene_store.frame_from_primary_secondary(p_axis, p_dir, s_axis, s_dir)
        note = f"primary {p_axis} = {p_desc}; secondary {s_axis} = {s_desc}" + (
            " (defaulted)" if s_defaulted else ""
        )
        prov = {"primary": {p_axis: p_name}, "secondary": {s_axis: s_name}}
    else:
        chosen: Dict[str, tuple] = {}
        for positive, negative in _AXIS_PAIRS:
            given = [k for k in (positive, negative) if orientation.get(k)]
            if len(given) > 1:
                raise BadRequest(
                    f"orientation gives both {positive} and {negative} — "
                    "they are the same axis; keep one of the two"
                )
            if given:
                chosen[positive] = (given[0], 1.0 if given[0] == positive else -1.0)
        if len(chosen) == 1:
            key = plain[0]
            if key in ("top", "bottom"):
                raise BadRequest(
                    f"a bare {key} direction cannot orient an upright "
                    "observer (upright already fixes that axis) — use the "
                    "primary/secondary form, e.g. someone lying face-up is "
                    '{"primary": {"back": "gravity"}, "secondary": '
                    '{"top": <head-end instance>}}'
                )
            if _is_bev_vector(orientation[key]):
                toward, axis_desc = await _bev_vector_dir(
                    str(orientation[key]).strip(), f"orientation.{key}"
                )
                axis_desc = f"along {axis_desc}"
                prov = {key: str(orientation[key]).strip()}
            else:
                toward, ref = _target_dir(orientation[key], f"orientation.{key}")
                axis_desc = f"toward {ref['desc']}"
                prov = {key: ref["name"]}
            down, up_src = await _gravity_dir()
            f, r, u = scene_store.frame_from_primary_secondary(
                "bottom", down, key, toward
            )
            note = f"upright ({up_src}), {key} {axis_desc}"
        elif len(chosen) == 2:
            raise BadRequest(
                "two bare axis directions do not say which one is exact — "
                'use the primary/secondary form: {"primary": {axis: value}, '
                '"secondary": {axis: value}} (value = instance name or '
                '"gravity")',
                got=sorted(k for k, _ in chosen.values()),
            )
        else:
            refs = {}
            dirs = {}
            for canonical, (key, sign) in chosen.items():
                d, ref = _target_dir(orientation[key], f"orientation.{key}")
                refs[canonical] = (key, ref)
                dirs[canonical] = d * sign
            f, r, u = scene_store.frame_from_targets(
                dirs["forward"], dirs["right"], dirs["top"]
            )
            note = (
                ", ".join(
                    f"{key} toward {ref['desc']}"
                    for key, ref in (refs[c] for c in ("forward", "right", "top"))
                )
                + " (orthonormalized, forward axis primary)"
            )
            prov = {refs[c][0]: refs[c][1]["name"] for c in ("forward", "right", "top")}
    axes = {
        "forward_w": [round(float(v), 6) for v in f],
        "right_w": [round(float(v), 6) for v in r],
        "up_w": [round(float(v), 6) for v in u],
    }
    return scene, axes, note, prov


async def _lift_position(
    position: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
    scene: Dict[str, Any],
    work_base: Path,
) -> tuple:
    """A new instance's world position from one of the three sources
    (at_camera / at_map / view+pixel): (pos_w, provenance, scene). Shared by
    /add_instance."""
    provenance: Dict[str, Any] = {}
    if position.get("at_camera") is not None:
        try:
            cam_frame = int(position["at_camera"])
        except (TypeError, ValueError) as exc:
            raise BadRequest("position.at_camera must be a frame index") from exc
        cam = registry.get(f"camera:f{cam_frame}")
        if cam is None:
            raise BadRequest(
                f"frame {cam_frame} is not among the reconstructed cameras",
                camera_frames=sorted(int(k) for k in scene.get("camera_poses") or {}),
            )
        pos_w = np.asarray(cam["position_w"], dtype=np.float64)
        provenance = {"at_camera": cam_frame}
    elif position.get("at_map") is not None:
        # A point read off the bev map: the most precise way to place an
        # observer once the map exists — pixel picks on wide-angle photos
        # land badly. Gated on the map actually having been BUILT for
        # that frame, so the coordinates come from looking at it, not
        # from imagination.
        spec = position["at_map"]
        try:
            m_frame = int(spec.get("frame"))
            mx, my = (float(v) for v in spec.get("pixel") or ())
        except (TypeError, ValueError) as exc:
            raise BadRequest(
                'position.at_map needs {"frame": N, "pixel": [PX, PY]} — '
                "pixel coordinates ON the frame-N bev map PNG (point at "
                "the map the way you point at any image)"
            ) from exc
        cam = registry.get(f"camera:f{m_frame}")
        if cam is None:
            raise BadRequest(f"frame {m_frame} is not among the reconstructed cameras")
        try:
            bev_env = _newest_bev_render(m_frame, work_base)
        except BadRequest:
            raise BadRequest(
                f"map pixels require the frame-{m_frame} bev map: run "
                f"bev --frame {m_frame} and LOOK at it first — PX,PY are "
                "pixel coordinates on that PNG"
            ) from None
        u_raw, v_raw = _bev_pixel_to_raw_uv(mx, my, bev_env, "position.at_map")
        scene = await _ensure_scene_gravity(scene, work_base)
        down = _scene_down(scene)
        fwd_g = np.asarray(cam["forward_w"], dtype=np.float64)
        fwd_g = fwd_g - down * float(fwd_g @ down)
        if float(np.linalg.norm(fwd_g)) < 0.2:
            fwd_g = np.asarray(cam["up_w"], dtype=np.float64)
            fwd_g = fwd_g - down * float(fwd_g @ down)
        fwd_g = fwd_g / np.linalg.norm(fwd_g)
        right_g = np.cross(down, fwd_g)
        right_g = right_g / np.linalg.norm(right_g)
        factor = scene_store.scale_factor(scene)
        cam_p = np.asarray(cam["position_w"], dtype=np.float64)
        horiz = cam_p + u_raw * right_g + v_raw * fwd_g
        pts_env = ctx.get("points_env")
        if pts_env is None:
            raise BadRequest("the scene has no reconstruction — run /recon first")
        cloud = np.asarray(pts_env["array"], dtype=np.float64).reshape(-1, 3)
        # Height comes from the cloud around the picked map pixel: static
        # points of every frame plus all points of the MAP's reference
        # frame — a mover passing through another frame's view must not
        # set this height.
        _pe = pts_env["array"]
        at_keep = _dynamic_flat_keep(
            ctx.get("dynamic_env"),
            [int(i) for i in pts_env["frame_indices"]],
            int(_pe.shape[1]),
            int(_pe.shape[2]),
            anchor_frame=m_frame,
        )
        if at_keep is not None:
            cloud = cloud[at_keep]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if cloud.shape[0] > _SCENE_CLOUD_MAX_PTS:
            cloud = cloud[:: cloud.shape[0] // _SCENE_CLOUD_MAX_PTS + 1]
        rel = cloud - horiz
        h_dist = np.linalg.norm(rel - np.outer(rel @ down, down), axis=1) * factor
        near = cloud[h_dist <= 0.30]
        if near.shape[0] < 30:
            near = cloud[h_dist <= 0.60]
        if near.shape[0] < 10:
            raise BadRequest(
                f"map pixel ({mx:g}, {my:g}) lands outside the reconstructed "
                "area (no 3D points within 0.6 m of it on the ground) — pick "
                "a point on the map's occupied region"
            )
        vert = float(np.median((near - cam_p) @ down))
        pos_w = horiz + down * vert
        provenance = {"at_map": {"frame": m_frame, "pixel": [mx, my]}}
    else:
        try:
            view = int(position.get("view"))
            px, py = (int(v) for v in position.get("pixel"))
        except (TypeError, ValueError) as exc:
            raise BadRequest(
                'position needs {"view": N, "pixel": [x, y]} (original-image '
                'pixels) or {"at_camera": N}'
            ) from exc
        points_env = ctx.get("points_env")
        if points_env is None:
            raise BadRequest("the scene has no reconstruction — run /recon first")
        points_frames = [int(i) for i in points_env["frame_indices"]]
        if view not in points_frames:
            raise BadRequest(
                f"view {view} is not among the reconstructed frames",
                frames=points_frames,
            )
        points = np.asarray(points_env["array"], dtype=np.float64)
        crop = points_env.get("crop_box") or {}
        cx = px - int(crop.get("left", 0))
        cy = py - int(crop.get("top", 0))
        grid_h, grid_w = int(points.shape[1]), int(points.shape[2])
        if not (0 <= cx < grid_w and 0 <= cy < grid_h):
            raise BadRequest(
                f"pixel ({px}, {py}) falls outside the reconstructed grid of "
                f"view {view}"
                + (
                    " (the reconstruction center-cropped this frame; pick a "
                    "pixel nearer the middle)"
                    if crop
                    else ""
                ),
                grid={"width": grid_w, "height": grid_h},
                crop_box=crop or None,
            )
        lf = points_frames.index(view)
        y0, y1 = max(0, cy - 2), min(grid_h, cy + 3)
        x0, x1 = max(0, cx - 2), min(grid_w, cx + 3)
        patch = points[lf, y0:y1, x0:x1].reshape(-1, 3)
        patch = patch[np.isfinite(patch).all(axis=1)]
        if patch.shape[0] == 0:
            raise BadRequest(
                f"no valid 3D points at pixel ({px}, {py}) of view {view} — "
                "the reconstruction has no depth there; pick a nearby pixel "
                "on the same surface"
            )
        pos_w = np.median(patch, axis=0)
        provenance = {"view": view, "pixel": [px, py]}
    return pos_w, provenance, scene


@app.post("/add_instance")
async def add_instance(request: Request) -> JSONResponse:
    """Register a named point in the scene, optionally with an orientation.

    Body: {"name": ..., "source": ...,
           "position": {"view": N, "pixel": [x, y]} | {"at_camera": N}
                     | {"at_map": {"frame": N, "pixel": [PX, PY]}},
           "orientation": {"facing": name} | {"facing_camera": N}
                        | one bare axis, e.g. {"forward": name}
                        | {"primary": {axis: value}, "secondary": {axis: value}}
                        | legacy three bare axes, one per pair}

    Position comes from the reconstruction: a pixel in a view is lifted to
    its 3D point (5x5 neighbourhood median), or a view's camera position is
    used directly. An orientation frame is a PRIMARY axis (taken exactly) +
    a SECONDARY direction (projected onto the plane orthogonal to it).
    Direction values are existing instances — objects, cameras (camera:fN),
    the room centre (scene) — the literal "gravity" (the downward world
    direction), or a bev map arrow "bev:f<N>:START:END". facing and
    single-bare-axis forms are upright sugar (primary bottom = gravity);
    e.g. someone lying face-up in bed is
    {"primary": {"back": "gravity"}, "secondary": {"top": <head-end>}}.

    Orientation can come LATER: a call naming an existing manual instance
    with an orientation and NO position orients it in place (register the
    spot, look at its marker on the bev map, then orient — often with an
    arrow read off that map).
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)

    name = str(payload.get("name") or "").strip()
    source = str(payload.get("source") or "").strip()
    if not name or len(name) > 64:
        raise BadRequest("name is required (a unique handle, at most 64 chars)")
    if not source:
        raise BadRequest(
            "source is required: one sentence saying how this instance "
            'entered the scene (e.g. "the door named in option C, pixel '
            'picked off frame_2")'
        )
    low = name.lower()
    if (
        low in ("scene",)
        or any(low.startswith(p) for p in _SCENE_RESERVED_PREFIXES)
        or "->" in name
        or "|" in name
    ):
        raise BadRequest(
            "that name is reserved (scene, camera:*, camplane:*) or "
            "contains reserved punctuation (->, |); pick another",
            got=name,
        )

    registry, ctx = _scene_registry(scene, work_base)
    position = payload.get("position")
    manual = scene.get("manual_instances") or {}
    if position is None and name in manual:
        # Orient (or re-orient) an EXISTING manual instance in place: the
        # two-step flow — register the spot, LOOK at its marker on the bev
        # map, then come back with just the orientation (often a map arrow).
        orientation = payload.get("orientation")
        if orientation is None:
            raise BadRequest(
                f"{name!r} already exists and no orientation was given — "
                "nothing to update. Give an orientation to orient it, or "
                "pick another name for a new instance"
            )
        pos_w = np.asarray(manual[name]["position_w"], dtype=np.float64)
        scene, axes, orientation_note, orient_prov = await _scene_orientation(
            orientation, pos_w, registry, ctx, scene, work_base
        )
        replaced = manual[name].get("forward_w") is not None

        def _orient(current: Dict[str, Any]) -> Dict[str, Any]:
            rec = (current.get("manual_instances") or {}).get(name)
            if rec is None:
                raise BadRequest(f"instance {name!r} vanished mid-update")
            rec.update(axes)
            prov = rec.setdefault("provenance", {})
            prov["orientation"] = orient_prov
            prov["orientation_source"] = source
            rec["updated"] = datetime.now(timezone.utc).isoformat()
            return current

        scene = _mutate_scene(work_base, _orient)
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "instance": {
                    "name": name,
                    "source": manual[name]["source"],
                    "has_orientation": True,
                    "orientation": orientation_note,
                },
                "position_note": "position unchanged (existing instance)",
                "units": _scene_units_note(scene),
                "scale_factor": round(scene_store.scale_factor(scene), 6),
                "note": (
                    ("orientation REPLACED" if replaced else "orientation added")
                    + " — re-run bev to check the facing arrow on the marker"
                ),
            },
        )
    if name in registry or any(n.lower() == low for n in registry):
        raise BadRequest(
            f"an instance named {name!r} already exists — names are unique "
            "handles; pick another or query the existing one",
            existing=sorted(registry),
        )
    if not isinstance(position, dict):
        raise BadRequest(
            'position is required for a NEW instance: {"view": N, "pixel": '
            '[x, y]} lifts a pixel of frame N to 3D; {"at_camera": N} uses '
            'that camera\'s position; {"at_map": {"frame": N, "pixel": [PX, PY]}} '
            "places it at that pixel of the frame-N bev map PNG. To orient an "
            "EXISTING instance, give its name plus an orientation and no "
            "position"
        )
    pos_w, provenance, scene = await _lift_position(
        position, registry, ctx, scene, work_base
    )

    orientation = payload.get("orientation")
    axes: Dict[str, Optional[List[float]]] = {
        "forward_w": None,
        "right_w": None,
        "up_w": None,
    }
    orientation_note = None
    if orientation is not None:
        scene, axes, orientation_note, orient_prov = await _scene_orientation(
            orientation, pos_w, registry, ctx, scene, work_base
        )
        provenance["orientation"] = orient_prov
    record = {
        "name": name,
        "source": source,
        "position_w": [round(float(v), 6) for v in pos_w],
        **axes,
        "provenance": provenance,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    def _apply(current: Dict[str, Any]) -> Dict[str, Any]:
        existing = current.setdefault("manual_instances", {})
        if name in existing:
            raise BadRequest(f"an instance named {name!r} already exists")
        existing[name] = record
        return current

    scene = _mutate_scene(work_base, _apply)
    factor = scene_store.scale_factor(scene)
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "instance": {
                "name": name,
                "source": source,
                "has_orientation": axes["forward_w"] is not None,
                **(
                    {"orientation": orientation_note}
                    if orientation_note
                    else {}
                ),
            },
            "position_note": (
                "position registered "
                + (
                    f"from view {provenance['view']} pixel {provenance['pixel']}"
                    if "view" in provenance
                    else (
                        f"from the frame-{provenance['at_map']['frame']} bev "
                        f"map at pixel ({provenance['at_map']['pixel'][0]:g}, "
                        f"{provenance['at_map']['pixel'][1]:g}) — height taken "
                        "from the reconstruction there"
                        if "at_map" in provenance
                        else f"at camera f{provenance['at_camera']}"
                    )
                )
            ),
            "units": _scene_units_note(scene),
            "scale_factor": round(factor, 6),
            "note": (
                "query it by name: distance/depth/compass all accept it; "
                "list_instance shows the full registry"
            ),
        },
    )


@app.post("/distance")
async def scene_distance(request: Request) -> JSONResponse:
    """Distance between two instances; optionally calibrate the scene scale.

    Body: {"a": name, "b": name} plus optional "known_distance": the
    question's stated distance between exactly these two instances — the
    scene records stated/raw as its scale factor, and every later
    distance, depth and map readout is reported in calibrated metres.
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    registry, ctx = _scene_registry(scene, work_base)
    a = _resolve_scene_point(payload.get("a"), registry, ctx, "a")
    b = _resolve_scene_point(payload.get("b"), registry, ctx, "b")
    delta = np.asarray(b["p3"], dtype=np.float64) - np.asarray(
        a["p3"], dtype=np.float64
    )
    raw = float(np.linalg.norm(delta))

    # Optional "mode": closest / farthest read the approach between the two
    # instances' reconstructed SURFACES (nearest / farthest pair of points,
    # outlier-trimmed) instead of the centre-to-centre figure — the
    # "measuring from the closest point of each object" reading. Absent
    # (or "centroid") the reply is exactly what it always was.
    mode = str(payload.get("mode") or "centroid").strip().lower()
    if mode not in ("centroid", "closest", "farthest"):
        raise BadRequest('mode must be "centroid" (default), "closest" or "farthest"')
    extremes = None
    centroid_raw = raw
    if mode != "centroid":
        try:
            extremes = scene_store.cloud_extremes(
                _instance_cloud(a), _instance_cloud(b)
            )
        except scene_store.SceneError as exc:
            raise BadRequest(str(exc)) from exc
        raw = float(extremes[mode])

    calibration = None
    if payload.get("known_distance") is not None:
        calibration = _apply_scene_calibration(
            work_base,
            "distance" if mode == "centroid" else f"distance ({mode} points)",
            a["desc"],
            b["desc"],
            payload["known_distance"],
            raw,
        )
        scene = _require_scene(work_base)

    factor = scene_store.scale_factor(scene)
    down = _scene_down(scene)
    if extremes is not None:
        single = [t["desc"] for t in (a, b) if (t["record"].get("pts") is None)]
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "a": a["desc"],
                "b": b["desc"],
                "mode": mode,
                "distance": round(raw * factor, 2),
                "centroid_distance": round(centroid_raw * factor, 2),
                "n_points": {"a": extremes["n_a"], "b": extremes["n_b"]},
                "units": _scene_units_note(scene),
                **({"calibration": calibration} if calibration else {}),
                **(
                    {
                        "single_point": (
                            ", ".join(single)
                            + " has no reconstructed surface (a camera or a "
                            "registered spot is one point), so its side of the "
                            "measurement is that point"
                        )
                    }
                    if single
                    else {}
                ),
                "note": (
                    (
                        "closest approach between the two instances' "
                        "reconstructed surfaces (nearest pair of points, "
                        "outlier-trimmed) — the reading for 'measuring from "
                        "the closest point of each object'"
                        if mode == "closest"
                        else "farthest separation between the two instances' "
                        "reconstructed surfaces (farthest pair of points, "
                        "outlier-trimmed)"
                    )
                    + "; centroid_distance is the plain centre-to-centre figure "
                    "for comparison"
                ),
            },
        )
    components = {}
    if down is not None:
        vertical = float(delta @ down)
        horizontal = float(np.linalg.norm(scene_store.horizontal(delta, down)))
        components = {
            "horizontal": round(horizontal * factor, 2),
            "vertical_downward": round(vertical * factor, 2) + 0.0,
        }
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "a": a["desc"],
            "b": b["desc"],
            "distance": round(raw * factor, 2),
            **({"components": components} if components else {}),
            "units": _scene_units_note(scene),
            **({"calibration": calibration} if calibration else {}),
            "note": (
                "straight-line 3D distance between the two instances' "
                "registered positions"
                + ("; components split it along/against gravity" if components else "")
            ),
        },
    )


def _instance_cloud(resolved: Dict[str, Any]) -> np.ndarray:
    """World points of a resolved instance: its segment cloud, or — for a
    camera / registered spot — its single registered position."""
    pts = resolved["record"].get("pts")
    if pts is None:
        return np.asarray(resolved["p3"], dtype=np.float64).reshape(1, 3)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


async def _scene_gravity_soft(
    scene: Dict[str, Any], work_base: Optional[Path]
) -> tuple:
    """Gravity if it can be had; (scene, down, warning-or-None). /size is
    useful without it (PCA box), so a calibration failure degrades instead
    of refusing."""
    try:
        scene = await _ensure_scene_gravity(scene, work_base)
    except GatewayError as exc:
        return (
            scene,
            None,
            f"gravity unavailable ({exc.message}); axes are not gravity-aligned",
        )
    return scene, _scene_down(scene), None


def _member_frames(rec: Dict[str, Any]) -> List[int]:
    frames = set()
    for m in rec.get("members") or []:
        head = str(m).partition(":")[0].lstrip("f")
        if head.isdigit():
            frames.add(int(head))
    return sorted(frames)


@app.post("/size")
async def scene_size(request: Request) -> JSONResponse:
    """Oriented bounding box of one or more segmented instances, merged.

    Body: {"of": [names]} — 1..8 segmented instances (registry names or
    f<N>:<label> per-frame references). Several names are MERGED into one
    cloud first (the same object registered twice from different views),
    then one gravity-aligned box is fitted: `height` along gravity,
    `length` >= `width` the minimum-area rectangle of the ground footprint.
    The reply names the longest of the three outright — the "longest
    dimension (length, width or height)" a size question asks for.
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    raw_targets = payload.get("of")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, list) or not (1 <= len(raw_targets) <= 8):
        raise BadRequest(
            '"of" must list 1..8 segmented instances (several = views of the '
            "same object, merged before the box is fitted)"
        )
    scene, down, gravity_warning = await _scene_gravity_soft(scene, work_base)
    registry, ctx = _scene_registry(scene, work_base)
    targets = [
        _resolve_scene_point(t, registry, ctx, f"of[{k}]")
        for k, t in enumerate(raw_targets)
    ]
    clouds = []
    frames: set = set()
    members: List[str] = []
    moving: List[str] = []
    for t in targets:
        rec = t["record"]
        if rec.get("pts") is None:
            raise BadRequest(
                f"{t['desc']!r} has no reconstructed surface (a camera or a "
                "registered spot is a single point) — size takes SEGMENTED "
                "instances; segment the object first",
                available=sorted(
                    n for n, r in registry.items() if r.get("pts") is not None
                ),
            )
        clouds.append(np.asarray(rec["pts"], dtype=np.float64).reshape(-1, 3))
        if rec.get("members"):
            frames.update(_member_frames(rec))
            members.extend(rec["members"])
        else:
            # an f<N>:<label> reference: one frame, named in the desc
            head = str(t["name"]).partition(":")[0].lstrip("f")
            if head.isdigit():
                frames.add(int(head))
            members.append(t["desc"].split(" ")[0])
        if rec.get("moving"):
            moving.append(t["desc"])
    merged = np.concatenate(clouds)
    try:
        box = scene_store.oriented_bbox(merged, down)
    except scene_store.SceneError as exc:
        raise BadRequest(str(exc)) from exc
    factor = scene_store.scale_factor(scene)
    ext = [float(v) * factor for v in box["extents"]]
    if box["gravity_aligned"]:
        axis_names = ["length", "width", "height"]
    else:
        axis_names = ["longest", "middle", "shortest"]
    k_max = int(np.argmax(ext))
    longest_m = ext[k_max]
    longest_cm = round(longest_m * 100.0)
    n_views = len(frames)
    desc = " + ".join(t["desc"] for t in targets)
    warnings: List[str] = []
    if gravity_warning:
        warnings.append(gravity_warning)
    if moving:
        warnings.append(
            f"{', '.join(moving)} is tagged MOVING: its views do not sit in "
            "one place, so a merged box is inflated by the motion — size a "
            "mover from ONE frame (f<N>:<label>)"
        )
    if n_views == 1:
        warnings.append(
            "seen from ONE view only: the side facing away from that camera is "
            "unobserved, so the extent along the viewing direction can read "
            "short — if the object appears in other frames, segment it there "
            "too and re-run size on the merged instance"
        )
    if box["n_points"] < 100:
        warnings.append(
            f"only {box['n_points']} reconstructed points — a small or distant "
            "object; treat the figure as rough"
        )
    dims = {name: round(v, 2) for name, v in zip(axis_names, ext)}
    conclusion = (
        f"{desc} measures "
        + " x ".join(f"{v:.2f}" for v in ext)
        + " m ("
        + " x ".join(axis_names)
        + (", gravity-aligned box" if box["gravity_aligned"] else ", PCA box")
        + f"); its longest dimension is its {axis_names[k_max].upper()}: "
        f"{longest_m:.2f} m = {longest_cm} cm"
    )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "of": [t["desc"] for t in targets],
            "merged": len(targets) > 1,
            "detections": sorted(set(members)),
            "n_views": n_views,
            "n_points": int(box["n_points"]),
            "dimensions": dims,
            "longest_dimension": {
                "axis": axis_names[k_max],
                "m": round(longest_m, 2),
                "cm": longest_cm,
            },
            "conclusion": conclusion,
            "units": _scene_units_note(scene),
            "method": (
                "oriented bounding box of the instance's reconstructed points: "
                "vertical axis = gravity, horizontal axes = minimum-area "
                "rectangle of the ground footprint; each extent is the 1-99 "
                "percentile span (mask edges bleed a few points onto the "
                "background, and a raw min/max would measure those)"
            ),
            **({"warnings": warnings} if warnings else {}),
        },
    )


def _scene_observer_and_targets(
    payload: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
):
    """Shared /depth and /side parsing: the observer (a camera view or an
    oriented instance), its unit axes, and the resolved targets."""
    if payload.get("view") is not None and payload.get("observer") is not None:
        raise BadRequest("give either view or observer, not both")
    if payload.get("view") is not None:
        try:
            observer_name = f"camera:f{int(payload['view'])}"
        except (TypeError, ValueError) as exc:
            raise BadRequest("view must be a frame index") from exc
    elif payload.get("observer") is not None:
        observer_name = str(payload["observer"])
    else:
        raise BadRequest(
            'an observer is required: {"view": N} for a camera, or '
            '{"observer": name} for an oriented instance'
        )
    obs = _resolve_scene_point(observer_name, registry, ctx, "observer")
    obs_rec = obs["record"]
    if obs_rec.get("forward_w") is None:
        raise BadRequest(
            f"instance {obs['desc']!r} has no orientation, so 'its view' is "
            "undefined — orient it in place: add_instance with the SAME "
            "name, no position, and an orientation (facing, one bare axis, "
            "or a primary/secondary pair — values are instance names, "
            '"gravity", or a bev map arrow "bev:f<N>:START:END"). '
            "The bev map shows HORIZONTAL directions around its marker — "
            "enough to pick an upright observer's heading; a non-upright "
            "pose reads its vertical axis from the pictures and the map "
            "only supplies horizontal parts (e.g. a lying person's head "
            "end)"
        )
    axes = {
        key: (
            scene_store.unit(np.asarray(obs_rec[key], dtype=np.float64))
            if obs_rec.get(key) is not None
            else None
        )
        for key in ("forward_w", "right_w", "up_w")
    }

    raw_targets = payload.get("of")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, list) or not raw_targets or len(raw_targets) > 8:
        raise BadRequest('"of" must list 1..8 instance names to read')
    targets = [
        _resolve_scene_point(t, registry, ctx, f"of[{k}]")
        for k, t in enumerate(raw_targets)
    ]
    return obs, axes, targets


@app.post("/depth")
async def scene_depth(request: Request) -> JSONResponse:
    """Depth of instances along an observer's forward axis — depth ONLY.

    Body: {"view": N, "of": [names]} — depth in camera f<N>'s view — or
    {"observer": name, "of": [names]} for any instance WITH an
    orientation. Optional "known_depth" (with exactly one target)
    calibrates the scene scale the same way /distance does. Positive
    depth = in front of the observer; negative = behind it. /side reads
    the full relation (front/behind, left/right, above/below) in one
    call; /depth is the specialist for known_depth calibration and
    nearest-to-farthest ordering.
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    registry, ctx = _scene_registry(scene, work_base)
    obs, axes, targets = _scene_observer_and_targets(payload, registry, ctx)
    obs_pos = np.asarray(obs["p3"], dtype=np.float64)
    fwd = axes["forward_w"]
    raw_depths = [
        float((np.asarray(t["p3"], dtype=np.float64) - obs_pos) @ fwd) for t in targets
    ]

    calibration = None
    if payload.get("known_depth") is not None:
        if len(targets) != 1:
            raise BadRequest(
                "known_depth calibrates against exactly ONE target's depth"
            )
        calibration = _apply_scene_calibration(
            work_base,
            "depth",
            obs["desc"],
            targets[0]["desc"],
            payload["known_depth"],
            raw_depths[0],
        )
        scene = _require_scene(work_base)

    factor = scene_store.scale_factor(scene)
    rows = [
        {
            "target": t["desc"],
            "depth": round(raw * factor, 2) + 0.0,
            **({"behind_observer": True} if raw < 0 else {}),
        }
        for t, raw in zip(targets, raw_depths)
    ]
    order_note = None
    if len(rows) >= 2:
        ranked = sorted(rows, key=lambda r: r["depth"])
        order_note = (
            "nearest to farthest along the observer's forward axis: "
            + " < ".join(f"{r['target']} ({r['depth']})" for r in ranked)
        )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "observer": obs["desc"],
            "depths": rows,
            **({"order": order_note} if order_note else {}),
            "units": _scene_units_note(scene),
            **({"calibration": calibration} if calibration else {}),
            "note": (
                "depth is the target's offset along the observer's forward "
                "axis, and ONLY that (positive = in front, negative = "
                "behind). It is not the straight-line distance (/distance); "
                "the full relation in this view (front/behind + left/right "
                "+ above/below) reads from /side"
            ),
        },
    )


@app.post("/side")
async def scene_side(request: Request) -> JSONResponse:
    """The full relation of instances in an observer's frame, stated
    outright: front/behind, left/right, above/below, plus the combined
    eight-way direction (front-left, rear-right, ...).

    Body mirrors /depth: {"view": N, "of": [names]} or {"observer": name,
    "of": [names]} for any oriented instance. For a CAMERA observer the
    frame is its gravity-aligned ground heading — the same frame the bev
    map draws — so a pitched camera cannot turn "behind on the floor
    plan" into "in front along the optical axis" (MMSI_333: a 34-deg-down
    camera read a cabinet 0.28 m behind it as 0.41 m "in front"). A
    registered observer keeps its own registered axes (a lying person's
    up is genuinely horizontal). /depth stays the depth specialist:
    known_depth calibration and nearest-to-farthest ordering.
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    registry, ctx = _scene_registry(scene, work_base)
    obs, axes, targets = _scene_observer_and_targets(payload, registry, ctx)
    if axes["right_w"] is None or axes["up_w"] is None:
        raise BadRequest(
            f"instance {obs['desc']!r} carries no right/up axes — re-add it "
            "with a full orientation"
        )
    camera_frame = obs["record"].get("kind") == "camera"
    if camera_frame:
        scene = await _ensure_scene_gravity(scene, work_base)
        axes = _ground_axes(axes, _scene_down(scene))
    obs_pos = np.asarray(obs["p3"], dtype=np.float64)
    factor = scene_store.scale_factor(scene)

    rows = []
    for t in targets:
        offset = np.asarray(t["p3"], dtype=np.float64) - obs_pos
        fwd_v = float(offset @ axes["forward_w"]) * factor
        right_v = float(offset @ axes["right_w"]) * factor
        up_v = float(offset @ axes["up_w"]) * factor
        depth_side = "front" if fwd_v > 0.05 else "behind" if fwd_v < -0.05 else "even"
        horizontal = (
            "right" if right_v > 0.05 else "left" if right_v < -0.05 else "in line"
        )
        vertical = "above" if up_v > 0.05 else "below" if up_v < -0.05 else "level"
        if depth_side == "even":
            d_text = f"even with {obs['desc']} along its forward axis"
        elif depth_side == "front":
            d_text = f"{fwd_v:.2f} m in FRONT of {obs['desc']}"
        else:
            d_text = f"{abs(fwd_v):.2f} m BEHIND {obs['desc']}"
        if horizontal == "in line":
            h_text = "laterally in line with its axis"
        else:
            h_text = f"{abs(right_v):.2f} m to its {horizontal.upper()}"
        if vertical == "level":
            v_text = "level with it"
        else:
            v_text = f"{abs(up_v):.2f} m {vertical.upper()} it"
        direction = None
        if math.hypot(fwd_v, right_v) >= 0.05:
            ang = math.degrees(math.atan2(right_v, fwd_v))
            octants = (
                "front",
                "front-right",
                "right",
                "rear-right",
                "rear",
                "rear-left",
                "left",
                "front-left",
            )
            direction = octants[int(((ang + 22.5) % 360.0) // 45.0)]
        conclusion = f"{t['desc']} is {d_text}, {h_text} and {v_text}"
        if direction is not None:
            conclusion += f" — overall to its {direction.upper()}"
        rows.append(
            {
                "target": t["desc"],
                "forward_m": round(fwd_v, 2) + 0.0,
                "right_m": round(right_v, 2) + 0.0,
                "up_m": round(up_v, 2) + 0.0,
                "front_or_behind": depth_side,
                "left_or_right": horizontal,
                "above_or_below": vertical,
                **({"direction": direction} if direction else {}),
                "conclusion": conclusion,
            }
        )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "observer": obs["desc"],
            "sides": rows,
            "units": _scene_units_note(scene),
            "note": (
                (
                    "the observer is a CAMERA, so the frame is its "
                    "gravity-aligned ground heading — the same frame the "
                    "bev map draws: positive forward = ahead of it on the "
                    "floor plan, positive right = to its right on the "
                    "floor plan, positive up = above it along gravity; "
                    "camera tilt cannot skew these. "
                    if camera_frame
                    else "signs are in the OBSERVER's registered frame: "
                    "positive forward = in front of it, positive right = "
                    "on its right, positive up = above it (a lying "
                    "person's up is genuinely horizontal). "
                )
                + "direction is the combined eight-way word from the "
                "forward/right components — when the options speak that "
                "language (rear left, front right...), it IS the answer; "
                "copy the conclusion, "
                "do not re-derive it from the pictures. Straight-line "
                "distance reads from /distance; ranking many targets "
                "nearest-to-farthest and known_depth calibration from "
                "/depth"
            ),
        },
    )


@app.post("/elevation")
async def scene_elevation(request: Request) -> JSONResponse:
    """Above/below along gravity — no observer involved.

    Body: {"of": [names]}, 2..8 instances; the FIRST is the reference the
    rest are read against. Scene gravity (GeoCalib, estimated once and
    persisted) gives the vertical, so each row is the target's signed
    height relative to the reference: positive up_m = above it. Camera
    tilt cannot skew this readout. /side's above_or_below is a different
    question — the OBSERVER's up axis, which tilts with the observer.
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    scene = await _ensure_scene_gravity(scene, work_base)
    down = _scene_down(scene)
    registry, ctx = _scene_registry(scene, work_base)
    raw_targets = payload.get("of")
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    if not isinstance(raw_targets, list) or not (2 <= len(raw_targets) <= 8):
        raise BadRequest(
            '"of" must list 2..8 instance names — the FIRST is the '
            "reference the others are read against"
        )
    targets = [
        _resolve_scene_point(t, registry, ctx, f"of[{k}]")
        for k, t in enumerate(raw_targets)
    ]
    factor = scene_store.scale_factor(scene)
    up = -down
    heights = [
        float(np.asarray(t["p3"], dtype=np.float64) @ up) * factor for t in targets
    ]
    ref = targets[0]
    rows = []
    for t, h in zip(targets[1:], heights[1:]):
        up_v = h - heights[0]
        verdict = "above" if up_v > 0.05 else "below" if up_v < -0.05 else "level"
        if verdict == "level":
            text = f"level with {ref['desc']} (within 0.05 m along gravity)"
        else:
            text = f"{abs(up_v):.2f} m {verdict.upper()} {ref['desc']}"
        rows.append(
            {
                "target": t["desc"],
                "up_m": round(up_v, 2) + 0.0,
                "above_or_below": verdict,
                "conclusion": f"{t['desc']} is {text}",
            }
        )
    order_note = None
    if len(targets) >= 3:
        ranked = sorted(zip(targets, heights), key=lambda pair: pair[1])
        order_note = "lowest to highest along gravity: " + " < ".join(
            f"{t['desc']} ({h - heights[0]:+.2f})" for t, h in ranked
        )
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "reference": ref["desc"],
            "elevations": rows,
            **({"order": order_note} if order_note else {}),
            "units": _scene_units_note(scene),
            "note": (
                "heights follow the scene's gravity, independent of every "
                "camera's tilt and of any observer; positive up_m = above "
                "the reference. The conclusion states the relation — copy "
                "it. An observer's own above/below (its up axis) reads "
                "from /side"
            ),
        },
    )


async def _scene_motion(payload: Dict[str, Any]) -> JSONResponse:
    """Scene /motion: {"of": [A, B]}, conclusion-style.

    Two cameras -> the relative pose between those instants (movement in
    the FIRST camera's gravity-aligned ground frame, turn about the
    vertical axis, sideways-lean change). The same object referenced in
    two frames (f<i>:X, f<j>:X) -> its shift and rotation via a trimmed
    rigid fit, the same math as the legacy /motion. Numbers live in the
    conclusion sentence — the reply is meant to be copied, not computed
    on."""
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)
    scene = await _ensure_scene_gravity(scene, work_base)
    down = _scene_down(scene)
    registry, ctx = _scene_registry(scene, work_base)
    raw = payload.get("of")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or len(raw) != 2:
        raise BadRequest(
            '"of" must list exactly TWO instances: two cameras (relative '
            "pose), or the same object seen in two frames (f<i>:X and "
            "f<j>:X)"
        )
    a = _resolve_scene_point(raw[0], registry, ctx, "of[0]")
    b = _resolve_scene_point(raw[1], registry, ctx, "of[1]")
    ra, rb = a["record"], b["record"]
    factor = scene_store.scale_factor(scene)

    def _ground_basis(rec: Dict[str, Any]):
        fwd = np.asarray(rec["forward_w"], dtype=np.float64)
        fwd = fwd - down * float(fwd @ down)
        if float(np.linalg.norm(fwd)) < 0.2:
            fwd = np.asarray(rec["up_w"], dtype=np.float64)
            fwd = fwd - down * float(fwd @ down)
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(down, fwd)
        return fwd, right / np.linalg.norm(right)

    oriented_pair = ra.get("kind") == rb.get("kind") == "literal" and (
        ra.get("forward_w") is not None and rb.get("forward_w") is not None
    )
    if (ra.get("kind") == "camera" and rb.get("kind") == "camera") or oriented_pair:
        fwd_g, right_g = _ground_basis(ra)
        delta = (
            np.asarray(rb["position_w"], dtype=np.float64)
            - np.asarray(ra["position_w"], dtype=np.float64)
        ) * factor
        du = float(delta @ right_g)
        dv = float(delta @ fwd_g)
        dh = -float(delta @ down)
        fwd_b = np.asarray(rb["forward_w"], dtype=np.float64)
        fwd_b = fwd_b - down * float(fwd_b @ down)
        yaw = None
        if float(np.linalg.norm(fwd_b)) >= 0.2:
            fwd_b = fwd_b / np.linalg.norm(fwd_b)
            yaw = math.degrees(math.atan2(float(fwd_b @ right_g), float(fwd_b @ fwd_g)))

        def _lean(rec: Dict[str, Any]) -> float:
            r = np.asarray(rec["right_w"], dtype=np.float64)
            return math.degrees(math.asin(float(np.clip(r @ down, -1.0, 1.0))))

        droll = _lean(rb) - _lean(ra)

        def _view_elev(rec: Dict[str, Any]) -> float:
            f = np.asarray(rec["forward_w"], dtype=np.float64)
            f = f / np.linalg.norm(f)
            return math.degrees(math.asin(float(np.clip(-(f @ down), -1.0, 1.0))))

        # Pitch — the reading MMSI_995 had no number for: "which way is
        # the camera rotating" with GT "down" found only movement, yaw and
        # roll in this reply, and the model guessed.
        dpitch = _view_elev(rb) - _view_elev(ra)
        moves = []
        if abs(dv) >= 0.10:
            moves.append(f"{abs(dv):.2f} m {'FORWARD' if dv > 0 else 'BACKWARD'}")
        if abs(du) >= 0.10:
            moves.append(f"{abs(du):.2f} m to its {'RIGHT' if du > 0 else 'LEFT'}")
        if abs(dh) >= 0.10:
            moves.append(f"{abs(dh):.2f} m {'UP' if dh > 0 else 'DOWN'}")
        parts = [
            "moved " + ", ".join(moves) if moves else "stayed in place (moved < 0.1 m)"
        ]
        if yaw is None:
            turn = None
            parts.append(
                "its ground heading is undefined (looking nearly straight up/down)"
            )
        elif abs(yaw) >= 3.0:
            turn = "clockwise" if yaw > 0 else "counterclockwise"
            side = "to its right" if yaw > 0 else "to its left"
            parts.append(f"turned {abs(yaw):.0f} deg {turn.upper()} ({side})")
        else:
            turn = "none"
            parts.append("kept its heading (turn < 3 deg)")
        if abs(dpitch) >= 3.0:
            parts.append(
                f"tilted its view {abs(dpitch):.0f} deg "
                f"{'UP' if dpitch > 0 else 'DOWN'}"
            )
        if abs(droll) >= 3.0:
            parts.append(
                f"its sideways lean changed by {droll:+.0f} deg (account for "
                "this before comparing angles between the two photos)"
            )
        cp_path = (scene.get("recon") or {}).get("camera_poses_path")
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "kind": "camera pair",
                "from": a["desc"],
                "to": b["desc"],
                **({"turn": turn} if turn is not None else {}),
                "conclusion": (
                    f"from {a['desc']} to {b['desc']} the camera " + "; ".join(parts)
                ),
                "units": _scene_units_note(scene),
                **(
                    {
                        "camera_poses": to_caller_path(
                            cp_path, caller_alias(payload), work_base=work_base
                        ),
                        "camera_poses_note": (
                            "the raw (N, 4, 4) camera-to-world extrinsics "
                            "(.npy; its .json sidecar's frame_indices gives "
                            "the row order). Read them ONLY when the "
                            "question itself sets up a frame to compute in "
                            "— its own axes, a rotation order to name; for "
                            "every plain direction question the conclusion "
                            "above IS the answer."
                        ),
                    }
                    if cp_path
                    else {}
                ),
                "note": (
                    "relative pose between the two camera instants: movement "
                    "is in the FIRST camera's gravity-aligned ground frame, "
                    "turn is about the vertical axis (clockwise = toward the "
                    "first camera's right, seen from above). Copy the "
                    "conclusion — do not re-derive camera motion from the "
                    "pictures"
                ),
            },
        )

    a_ref = re.fullmatch(r"[fF](\d+):.+", str(raw[0]).strip())
    b_ref = re.fullmatch(r"[fF](\d+):.+", str(raw[1]).strip())
    if a_ref and b_ref and ra.get("pts") is not None and rb.get("pts") is not None:
        fa, fb = int(a_ref.group(1)), int(b_ref.group(1))
        if fa == fb:
            raise BadRequest(
                "the two references name the SAME frame — object motion is "
                "the change between two frames (f<i>:X and f<j>:X, i != j)"
            )
        cam = registry.get(f"camera:f{fa}")
        if cam is None:
            raise BadRequest(f"frame {fa} is not among the reconstructed cameras")
        fwd_g, right_g = _ground_basis(cam)

        def _coords(pts: np.ndarray) -> np.ndarray:
            pw = np.asarray(pts, dtype=np.float64)
            pw = pw[np.isfinite(pw).all(axis=1)]
            if len(pw) > 1500:
                pw = pw[:: len(pw) // 1500 + 1]
            if len(pw) < 50:
                raise BadRequest(
                    f"only {len(pw)} usable 3D points on one side — too few "
                    "to fit a rigid motion"
                )
            return np.stack([pw @ right_g, pw @ fwd_g, -(pw @ down)], axis=1)

        A = _coords(ra["pts"])
        B = _coords(rb["pts"])
        R, _t, _rms, _n = _rigid_icp(A, B)
        dtr = (B.mean(0) - A.mean(0)) * factor
        du, dv, dh = (float(x) for x in dtr)
        dist = math.hypot(du, dv)
        e = R @ np.array([0.0, 1.0, 0.0])
        yaw = math.degrees(math.atan2(float(e[0]), float(e[1])))
        w = R @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(float(np.clip(w[2], -1.0, 1.0))))
        # Shift significance is a PERCENTAGE of the object's own size (a
        # mug shifted 8 cm moved; a sofa shifted 8 cm did not), between an
        # absolute noise floor and an absolute cap: robust diameter = 2 x
        # the 85th-percentile radius about the centroid, averaged over the
        # two views.
        size = float(
            np.percentile(np.linalg.norm(A - A.mean(0), axis=1), 85)
            + np.percentile(np.linalg.norm(B - B.mean(0), axis=1), 85)
        ) * factor
        shift_gate = min(
            max(_MOTION_SHIFT_FLOOR_M, _MOTION_SHIFT_FRAC * size),
            _MOTION_SHIFT_CAP_M,
        )
        parts = []
        if dist >= shift_gate or abs(dh) >= shift_gate:
            # Direction, not just magnitude: MMSI_981 got "shifted 1.09 m"
            # and "1.39 m" for two frame pairs and could not order the
            # frames from magnitudes alone — it fell back to eyeballing
            # pixel columns. Components are in the from-frame camera's
            # ground view, same convention as the camera-pair reading.
            parts.append(
                f"shifted {dist:.2f} m on the ground (seen from camera "
                f"f{fa}: {abs(du):.2f} m {'RIGHT' if du > 0 else 'LEFT'}, "
                f"{abs(dv):.2f} m {'AWAY' if dv > 0 else 'TOWARD'}) and "
                f"{dh:+.2f} m vertically"
            )
        rotation = "none"
        if abs(yaw) >= _MOTION_ROT_DEG:
            rotation = "clockwise" if yaw > 0 else "counterclockwise"
            parts.append(
                f"rotated {abs(yaw):.0f} deg {rotation.upper()} seen from above"
            )
        if tilt >= _MOTION_ROT_DEG:
            parts.append(f"tilted {tilt:.0f} deg off vertical")
        # In-image rotation: the SAME R projected onto the from-frame
        # camera's viewing axis — what a "rotated clockwise or counter-
        # clockwise (in the picture)" question actually asks. yaw misses
        # it whenever the physical axis is near horizontal: MMSI_376's pan
        # turned ~8 deg about an almost-horizontal axis — +4.9 deg about
        # the viewing axis (clockwise, the answer) but only -3.5 deg of
        # yaw. Positive about the viewing direction = clockwise on the
        # image; reading the object's WORLD rotation keeps it immune to
        # camera roll (the apparent between-photo difference is not).
        image_rotation = "none"
        image_rotation_deg = None
        if cam.get("forward_w") is not None:
            rvec, _ = cv2.Rodrigues(R)
            z_w = np.asarray(cam["forward_w"], dtype=np.float64)
            z_g = np.array(
                [float(z_w @ right_g), float(z_w @ fwd_g), -float(z_w @ down)]
            )
            z_norm = float(np.linalg.norm(z_g))
            if z_norm > 1e-6:
                comp = math.degrees(float(rvec.ravel() @ (z_g / z_norm)))
                image_rotation_deg = comp
                if abs(comp) >= _MOTION_ROT_DEG:
                    image_rotation = (
                        "clockwise" if comp > 0 else "counterclockwise"
                    )
                    parts.append(
                        f"seen from camera f{fa}, rotated {abs(comp):.0f} deg "
                        f"{image_rotation.upper()} in the image plane"
                    )
        conclusion = (
            f"{a['desc']} -> {b['desc']} " + "; ".join(parts)
            if parts
            else (
                f"{a['desc']} -> {b['desc']}: no significant motion detected "
                "in the reconstruction"
            )
        )
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "kind": "object pair",
                "from": a["desc"],
                "to": b["desc"],
                "rotation": rotation,
                "image_rotation": image_rotation,
                "conclusion": conclusion,
                "units": _scene_units_note(scene),
                "note": (
                    "trimmed rigid fit between the object's reconstructed "
                    "points in the two frames; 'rotation' is about the "
                    "vertical axis, seen from above; 'image_rotation' is "
                    "the same turn as camera f<from> sees it — THE reading "
                    "for clockwise/counterclockwise-in-the-picture "
                    "questions, and immune to camera roll. The split-pair "
                    "bev maps (bev --erase) are the VISUAL check. HONEST "
                    "LIMIT: depth models often reconstruct dark or "
                    "textureless objects as static — when this says static "
                    "but the photos clearly show the object being held or "
                    "manipulated, judge the motion from the photos and say "
                    "so"
                ),
            },
        )


    raise BadRequest(
        '"of" pairs two cameras (relative pose) or the same object seen in '
        "two frames (f<i>:X and f<j>:X); an object's relation to an "
        "observer reads from /side"
    )


def _scene_compass_line(
    ref_desc: str,
    p2: np.ndarray,
    d2: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> Dict[str, Any]:
    return {
        "kind": "line",
        "p": p2,
        "d": d2,
        "t_lo": t_lo,
        "t_hi": t_hi,
        "desc": ref_desc,
    }


def _resolve_scene_compass_anchor(
    text: Any,
    want_plane: bool,
    registry: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
    e1: np.ndarray,
    e2: np.ndarray,
    factor: float,
    field: str,
) -> Dict[str, Any]:
    """Ground an anchor on the horizontal plane: a point or a line (plane).

    Grammar matches the scene readouts plus the two camera forms:
    instance name / f<N>:<label-or-id> (point; "|plane" handled by the
    caller via want_plane), camplane:f<N> (always a plane), and any
    oriented instance can also anchor by plane of its own footprint.
    """
    raw = str(text or "").strip()
    low = raw.lower()
    if low.startswith("camplane:"):
        tail = low.removeprefix("camplane:").removeprefix("f")
        if not tail.isdigit():
            raise BadRequest(f'{field}: camplane takes "camplane:f<N>"', got=raw)
        cam = registry.get(f"camera:f{int(tail)}")
        if cam is None:
            raise BadRequest(
                f"{field}: frame {tail} is not among the reconstructed cameras"
            )
        fwd2 = scene_store.to_ground(cam["forward_w"], e1, e2)
        norm = float(np.hypot(fwd2[0], fwd2[1]))
        if norm < 0.15:
            raise BadRequest(
                f"{field}: camera f{int(tail)} looks nearly straight up/"
                "down — its image plane has no direction on the ground plane"
            )
        fwd2 = fwd2 / norm
        p2 = scene_store.to_ground(cam["position_w"], e1, e2)
        return _scene_compass_line(
            f"camera f{int(tail)} image plane",
            p2,
            np.array([fwd2[1], -fwd2[0]]),
            -1.2,
            1.2,
        )
    ref = _resolve_scene_point(raw, registry, ctx, field)
    p2 = scene_store.to_ground(ref["p3"], e1, e2)
    if not want_plane:
        return {"kind": "point", "p": p2, "desc": ref["desc"]}
    pts = (ref.get("record") or {}).get("pts")
    if pts is None:
        raise BadRequest(
            f"{field}: {ref['desc']} has no measured footprint to fit a "
            "plane through — planes belong to segmented wall-like "
            "structure; use the centroid (drop the plane flag)"
        )
    down = ctx.get("down")
    p0, axis, t_lo, t_hi = scene_store.ground_line_fit(pts, down)
    if (t_hi - t_lo) * factor < 0.5:
        raise BadRequest(
            f"{field}: {ref['desc']} spans only {(t_hi - t_lo) * factor:.2f} m "
            "of ground — too small to define a plane direction; small "
            "objects are points: use the centroid instead (drop the plane "
            "flag)"
        )
    return _scene_compass_line(ref["desc"] + " plane", p0, axis, t_lo - 0.2, t_hi + 0.2)


def _scene_compass_arrow(a: Dict[str, Any], b: Dict[str, Any], where: str):
    def _foot(point, line):
        rel = point - line["p"]
        return line["p"] + float(rel @ line["d"]) * line["d"]

    if a["kind"] == "line" and b["kind"] == "line":
        raise BadRequest(
            f"{where}: start and end cannot both be planes; at least one "
            "side must be a point (a point-to-plane arrow runs along the "
            "perpendicular)"
        )
    if a["kind"] == "line":
        e_pt = b["p"]
        s_pt = _foot(e_pt, a)
    elif b["kind"] == "line":
        s_pt = a["p"]
        e_pt = _foot(s_pt, b)
    else:
        s_pt, e_pt = a["p"], b["p"]
    vec = e_pt - s_pt
    if float(np.hypot(vec[0], vec[1])) < 0.05:
        raise BadRequest(
            f"{where}: start and end land on the same ground point "
            "(< 0.05 m apart) — the direction is undefined; use anchors "
            "separated on the ground",
            start=a["desc"],
            end=b["desc"],
        )
    return s_pt, e_pt, vec


@app.post("/compass")
async def scene_compass(request: Request) -> JSONResponse:
    """Compass over the scene: anchor once, query between instances forever.

    Body: optional anchor {"start": ref, "end": ref, "orientation": dir,
    "start_plane"/"end_plane": bool} declaring that the direction from
    start to end IS that compass orientation, plus optional
    "queries": [{"start": ref, "end": ref, ...}]. The orientation fact is
    persisted IN THE SCENE (world frame): any later /compass or /bev call
    reuses it without re-anchoring, whatever the reference frame. Refs are
    instance names (see /list_instance), f<N>:<label-or-id>, or
    camplane:f<N>; gravity comes from the scene (GeoCalib, best frames).
    """
    payload = await read_json_body(request)
    work_base = resolve_work_base(payload)
    scene = _require_scene(work_base)

    anchor_given = any(
        payload.get(k) is not None for k in ("start", "end", "orientation")
    )
    raw_queries = payload.get("queries")
    if raw_queries is not None and (
        not isinstance(raw_queries, list) or len(raw_queries) > 8
    ):
        raise BadRequest('queries must be a list (max 8) of {"start", "end"} objects')
    if not anchor_given and not raw_queries:
        raise BadRequest(
            "compass needs an anchor (start/end/orientation) and/or "
            "queries — an empty compass call does nothing"
        )

    scene = await _ensure_scene_gravity(scene, work_base)
    down = _scene_down(scene)
    e1, e2 = scene_store.horizontal_basis(down)
    registry, ctx = _scene_registry(scene, work_base)
    factor = scene_store.scale_factor(scene)

    def _resolve(text, want_plane, field):
        return _resolve_scene_compass_anchor(
            text, want_plane, registry, ctx, e1, e2, factor, field
        )

    anchor_block = None
    if anchor_given:
        for k in ("start", "end", "orientation"):
            if payload.get(k) is None:
                raise BadRequest(
                    "an anchor declaration needs all three of start/end/"
                    "orientation (the direction from start to end IS that "
                    "orientation)"
                )
        orient = _parse_compass_orientation(payload.get("orientation"))
        start = _resolve(payload["start"], bool(payload.get("start_plane")), "start")
        end = _resolve(payload["end"], bool(payload.get("end_plane")), "end")
        s_pt, e_pt, vec = _scene_compass_arrow(start, end, "compass anchor")
        north2 = scene_store.north_2d_from_arrow(vec, _COMPASS_OFFSET_DEG[orient])
        north_world = north2[0] * e1 + north2[1] * e2
        summary = f"{start['desc']} -> {end['desc']} is {orient}"
        anchor_record = {
            "summary": summary,
            "start": str(payload["start"]),
            "end": str(payload["end"]),
            "orientation": orient,
            "at": datetime.now(timezone.utc).isoformat(),
        }

        def _apply(current: Dict[str, Any]) -> Dict[str, Any]:
            compass = current.get("compass") or {"history": []}
            if compass.get("north_world") is not None:
                anchor_record["replaced"] = compass.get("summary")
            compass["north_world"] = [round(float(v), 6) for v in north_world]
            compass["summary"] = summary
            compass.setdefault("history", []).append(anchor_record)
            current["compass"] = compass
            return current

        scene = _mutate_scene(work_base, _apply)
        anchor_block = {
            "start_grounded": start["desc"],
            "end_grounded": end["desc"],
            "orientation": orient,
            "summary": summary,
            **(
                {"replaced": anchor_record["replaced"]}
                if "replaced" in anchor_record
                else {}
            ),
        }
    else:
        stored = scene.get("compass") or {}
        if stored.get("north_world") is None:
            raise BadRequest(
                "compass queries need an anchored compass — declare "
                "start/end/orientation from the ONE orientation fact the "
                "question states (in this call or an earlier one) first"
            )
        north_world = np.asarray(stored["north_world"], dtype=np.float64)
        north2 = scene_store.to_ground(north_world, e1, e2)
        norm = float(np.hypot(north2[0], north2[1]))
        if norm < 1e-6:
            raise BadRequest(
                "the stored compass has no horizontal component under the "
                "current gravity — re-anchor it"
            )
        north2 = north2 / norm

    def _diff(x: float, y: float) -> float:
        d = abs(x - y) % 360.0
        return min(d, 360.0 - d)

    query_rows = []
    for k, rq in enumerate(raw_queries or []):
        if not isinstance(rq, dict):
            raise BadRequest(f"queries[{k}] must be an object")
        qa = _resolve(
            rq.get("start"), bool(rq.get("start_plane")), f"queries[{k}].start"
        )
        qb = _resolve(rq.get("end"), bool(rq.get("end_plane")), f"queries[{k}].end")
        _s, _e, vec = _scene_compass_arrow(qa, qb, f"queries[{k}]")
        qbear = scene_store.bearing_deg_2d(vec, north2)
        ranked = sorted(
            _COMPASS_ORIENTATIONS,
            key=lambda n: _diff(qbear, _COMPASS_OFFSET_DEG[n]),
        )
        nearest = [
            {
                "orientation": n,
                "diff_deg": round(_diff(qbear, _COMPASS_OFFSET_DEG[n]), 1),
            }
            for n in ranked
        ]
        margin = round(nearest[1]["diff_deg"] - nearest[0]["diff_deg"], 1)
        top, runner = nearest[0], nearest[1]
        conclusion = (
            f"{qa['desc']} -> {qb['desc']} points "
            f"{top['orientation'].upper()} "
            f"({top['diff_deg']:.1f} deg off; next candidate "
            f"{runner['orientation']} is {runner['diff_deg']:.1f} deg off)"
        )
        if margin < 10.0:
            conclusion += (
                f" — a close call: the top two are within {margin:.1f} deg "
                "of each other"
            )
        side_seen = cam_i = obj_desc = None
        if qa["kind"] == "point" and qb["desc"].endswith("image plane"):
            cam_i = qb["desc"].split(" ")[1]
            side_seen = top["orientation"]
            obj_desc = qa["desc"]
        elif qa["desc"].endswith("image plane") and qb["kind"] == "point":
            cam_i = qa["desc"].split(" ")[1]
            rbear = (qbear + 180.0) % 360.0
            side_seen = min(
                _COMPASS_ORIENTATIONS,
                key=lambda n: _diff(rbear, _COMPASS_OFFSET_DEG[n]),
            )
            obj_desc = qb["desc"]
        if side_seen is not None:
            conclusion += (
                f"; side-seen rule: view {cam_i} sees the "
                f"{obj_desc}'s {side_seen.upper()} side"
            )
        query_rows.append(
            {
                "start": str(rq.get("start")),
                "end": str(rq.get("end")),
                "start_grounded": qa["desc"],
                "end_grounded": qb["desc"],
                "bearing_deg": round(qbear, 1) % 360.0,
                "direction": top["orientation"],
                "conclusion": conclusion,
                "nearest": nearest,
                "margin_deg": margin,
                **(
                    {
                        "ambiguous_note": (
                            "the two nearest directions are within 10 deg "
                            "of each other (see margin_deg); reported so "
                            "the call can be described as close — the "
                            "ranking above still stands"
                        )
                    }
                    if margin < 10.0
                    else {}
                ),
            }
        )

    stored = scene.get("compass") or {}
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            **({"anchor": anchor_block} if anchor_block else {}),
            **(
                {"restored": True, "anchored_by": stored.get("summary")}
                if not anchor_block
                else {}
            ),
            "mappings": [h.get("summary") for h in stored.get("history") or []]
            or ([anchor_block["summary"]] if anchor_block else []),
            **({"queries": query_rows} if query_rows else {}),
            **(
                {
                    "queries_note": (
                        "each query is the compass direction FROM its start "
                        "TO its end (point-to-plane runs along the "
                        "perpendicular); 'direction'/'conclusion' ARE the "
                        "tool's answer — report that direction; 'nearest' "
                        "ranks ALL EIGHT compass directions by angular "
                        "distance as the supporting evidence — a direction "
                        "far down the list cannot be the answer, no matter "
                        "what the images suggest"
                    )
                }
                if query_rows
                else {}
            ),
            "note": (
                "the orientation mapping is persisted in the scene: later "
                "compass calls query directly without re-anchoring, and "
                "later bev maps draw the rose from it"
            ),
        },
    )


@app.post("/measure")
async def measure(request: Request) -> JSONResponse:
    """Numeric summaries over saved arrays. Pure numpy, no GPU work.

    ops:
      distance  {"op":"distance","point_a":[x,y,z],"point_b":[x,y,z]}
                or two mask centroids via a/b objects (see client_examples.md)
      centroid  {"op":"centroid","points":"...npy","masks":"...npy",
                 "frame_index":0,"instance":0}
      bbox3d    same inputs as centroid
    """
    payload = await read_json_body(request)
    op = payload.get("op")
    work_base = resolve_work_base(payload)

    # "depth_at" closes the affordance gap that showed up in the first ERQA run:
    # agents were handed an absolute host path to a depth/points .npy and tried
    # `np.load(...)` inside the container, where that path does not exist
    # (artifacts live on the gateway host only). Every such attempt burned a step
    # on FileNotFoundError. Reading a value at a pixel is the query they were
    # actually reaching for, so serve it here instead of exposing the array.
    # "region_stats" answers the question the ascii-art pixel dumps were groping
    # toward. In the first ERQA-100 run, 78% of trajectories printed character
    # grids of pixel values (100% of the failures did) because a text-only scaffold
    # can only get an image back as characters. Serving aggregate statistics over a
    # rectangle is cheap, exact, and costs one step instead of dozens.
    if op == "region_stats":
        points_env = load_envelope(payload.get("points"), "points", work_base=work_base)
        frames = [int(i) for i in points_env["frame_indices"]]
        frame_index = int(payload.get("frame_index", frames[0] if frames else 0))
        if frame_index not in frames:
            raise BadRequest(
                f"frame_index {frame_index} not in this array", available=frames
            )
        arr = points_env["array"]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise BadRequest(
                "points array must have shape (N, H, W, 3)", shape=list(arr.shape)
            )
        fp = arr[frames.index(frame_index)]
        h, w = fp.shape[:2]
        box = payload.get("box", [0, 0, w, h])
        try:
            x0, y0, x1, y1 = (int(v) for v in box)
        except (TypeError, ValueError):
            raise BadRequest("box must be [x0, y0, x1, y1]", got=repr(box))
        x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
        x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
        sub = fp[y0:y1, x0:x1].reshape(-1, 3)
        finite = np.isfinite(sub).all(axis=1)
        valid = sub[finite]
        if valid.size == 0:
            return JSONResponse(
                content={
                    "op": "region_stats",
                    "frame_index": frame_index,
                    "box": [x0, y0, x1, y1],
                    "n_valid": 0,
                    "message": (
                        "No finite 3D points in that rectangle: the reconstruction did "
                        "not resolve this region. Try a larger box or a different frame."
                    ),
                }
            )
        d = valid[:, 2]
        pct = [float(np.percentile(d, p)) for p in (5, 25, 50, 75, 95)]
        return JSONResponse(
            content={
                "op": "region_stats",
                "frame_index": frame_index,
                "box": [x0, y0, x1, y1],
                "n_valid": int(valid.shape[0]),
                "n_total": int(sub.shape[0]),
                "depth_m": {
                    "min": round(float(d.min()), 6),
                    "max": round(float(d.max()), 6),
                    "mean": round(float(d.mean()), 6),
                    "p5": round(pct[0], 6),
                    "p25": round(pct[1], 6),
                    "median": round(pct[2], 6),
                    "p75": round(pct[3], 6),
                    "p95": round(pct[4], 6),
                },
                "centroid_xyz_m": np.round(valid.mean(axis=0), 6).tolist(),
                "message": (
                    "Aggregate 3D statistics over the rectangle. Use this instead of "
                    "printing pixel grids: it is exact and costs one call."
                ),
            }
        )

    if op == "depth_at":
        points_env = load_envelope(payload.get("points"), "points", work_base=work_base)
        frames = [int(i) for i in points_env["frame_indices"]]
        frame_index = int(payload.get("frame_index", frames[0] if frames else 0))
        if frame_index not in frames:
            raise BadRequest(
                f"frame_index {frame_index} not in this array",
                available=frames,
            )
        arr = points_env["array"]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise BadRequest(
                "points array must have shape (N, H, W, 3)", shape=list(arr.shape)
            )
        frame_points = arr[frames.index(frame_index)]
        h, w = frame_points.shape[:2]
        pixels = payload.get("pixels")
        if not isinstance(pixels, list) or not pixels:
            raise BadRequest(
                "depth_at requires pixels: [[x, y], ...] in image coordinates",
                got=repr(pixels),
            )
        out = []
        for i, px in enumerate(pixels):
            try:
                x, y = int(px[0]), int(px[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise BadRequest(f"pixels[{i}] must be [x, y]", got=repr(px)) from exc
            if not (0 <= x < w and 0 <= y < h):
                raise BadRequest(
                    f"pixels[{i}] out of bounds",
                    got=[x, y],
                    image_size={"width": w, "height": h},
                )
            xyz = frame_points[y, x]
            finite = bool(np.all(np.isfinite(xyz)))
            out.append(
                {
                    "pixel": [x, y],
                    "xyz_m": [float(v) for v in xyz] if finite else None,
                    "depth_m": float(xyz[2]) if finite else None,
                    "valid": finite,
                }
            )
        n_valid = sum(1 for r in out if r["valid"])
        return JSONResponse(
            content={
                "op": "depth_at",
                "frame_index": frame_index,
                "image_size": {"width": w, "height": h},
                "points": out,
                "n_valid": n_valid,
                "message": (
                    "depth_m is metric distance along the camera's viewing axis; xyz_m is the "
                    "3D point in the reconstruction frame. valid=false means the reconstruction "
                    "has no finite estimate at that pixel (common on sky, glass and thin edges) "
                    "— sample a nearby pixel rather than treating it as an error."
                    if n_valid < len(out)
                    else "depth_m is metric distance along the camera's viewing axis; xyz_m is the "
                    "3D point in the reconstruction frame."
                ),
            }
        )

    if op not in {"distance", "centroid", "bbox3d", "depth_at"}:
        raise BadRequest(
            "op must be one of: distance, centroid, bbox3d, depth_at", got=repr(op)
        )

    def resolve_region(spec: Dict[str, Any], field: str) -> Dict[str, Any]:
        """A region is either a literal 3D point or a (points, masks) pair."""
        if not isinstance(spec, dict):
            raise BadRequest(f"{field} must be a JSON object")
        if "point" in spec:
            point = coerce_point3(spec["point"], f"{field}.point")
            return {
                "centroid": point,
                "n_points": 1,
                "frame_indices": None,
                "frame_index": None,
            }
        points_env = load_envelope(
            spec.get("points"), f"{field}.points", work_base=work_base
        )
        masks_env = load_envelope(
            spec.get("masks"), f"{field}.masks", work_base=work_base
        )
        conf_env = (
            load_envelope(
                spec.get("confidence"), f"{field}.confidence", work_base=work_base
            )
            if spec.get("confidence")
            else None
        )
        try:
            frame_index = int(
                spec.get("frame_index", (points_env["frame_indices"] or [0])[0])
            )
            instance = int(spec.get("instance", 0))
            min_conf = float(spec.get("min_confidence", 0.0))
        except (TypeError, ValueError) as exc:
            raise BadRequest(
                f"{field}: frame_index/instance/min_confidence must be numbers"
            ) from exc
        pts = _mask_points(
            points_env, masks_env, frame_index, instance, conf_env, min_conf
        )
        common = {
            "frame_indices": [int(i) for i in masks_env["frame_indices"]],
            "frame_index": frame_index,
        }
        if pts.shape[0] == 0:
            return {"centroid": None, "n_points": 0, **common}
        return {
            "centroid": pts.mean(axis=0),
            "points": pts,
            "n_points": int(pts.shape[0]),
            **common,
        }

    if op == "distance":
        if "point_a" in payload or "point_b" in payload:
            region_a = {
                "centroid": coerce_point3(payload.get("point_a"), "point_a"),
                "n_points": 1,
                "frame_indices": None,
                "frame_index": None,
            }
            region_b = {
                "centroid": coerce_point3(payload.get("point_b"), "point_b"),
                "n_points": 1,
                "frame_indices": None,
                "frame_index": None,
            }
        else:
            region_a = resolve_region(payload.get("a") or {}, "a")
            region_b = resolve_region(payload.get("b") or {}, "b")
            # A distance is only meaningful between two regions measured on the
            # same frame, unless the caller opts in to a cross-frame distance
            # (valid for a static scene in a shared world frame).
            frame_a, frame_b = region_a["frame_index"], region_b["frame_index"]
            allow_cross = bool(payload.get("allow_cross_frame", False))
            if (
                frame_a is not None
                and frame_b is not None
                and frame_a != frame_b
                and not allow_cross
            ):
                raise FrameMismatch(
                    "Frame index mismatch between the two measured regions: "
                    f"region a was measured on frame {frame_a} and region b on "
                    f"frame {frame_b}. Measure both on the same frame, or pass "
                    f'"allow_cross_frame": true if you intend a world-space '
                    f"distance across frames of a static scene.",
                    a_frame_index=frame_a,
                    b_frame_index=frame_b,
                    a_frame_indices=region_a["frame_indices"],
                    b_frame_indices=region_b["frame_indices"],
                )
        if region_a["centroid"] is None or region_b["centroid"] is None:
            return JSONResponse(
                status_code=200,
                content={
                    "found": False,
                    "op": "distance",
                    "message": (
                        "One of the regions had no valid 3D points (empty mask or "
                        "all-nonfinite depth). No distance could be measured."
                    ),
                    "n_points_a": region_a["n_points"],
                    "n_points_b": region_b["n_points"],
                },
            )
        delta = np.asarray(region_a["centroid"]) - np.asarray(region_b["centroid"])
        return JSONResponse(
            status_code=200,
            content={
                "found": True,
                "op": "distance",
                "distance_m": round(float(np.linalg.norm(delta)), 6),
                "delta_xyz_m": np.round(delta, 6).tolist(),
                "point_a": np.round(
                    np.asarray(region_a["centroid"], float), 6
                ).tolist(),
                "point_b": np.round(
                    np.asarray(region_b["centroid"], float), 6
                ).tolist(),
                "n_points_a": region_a["n_points"],
                "n_points_b": region_b["n_points"],
                "frame_indices": [
                    f
                    for f in (region_a["frame_index"], region_b["frame_index"])
                    if f is not None
                ],
                "units": "metres",
            },
        )

    region = resolve_region(payload, "request")
    if region["centroid"] is None:
        return JSONResponse(
            status_code=200,
            content={
                "found": False,
                "op": op,
                "n_points": 0,
                "message": (
                    "The mask selected no points with finite 3D coordinates. "
                    "This is an observation, not a failure."
                ),
            },
        )

    if op == "centroid":
        return JSONResponse(
            status_code=200,
            content={
                "found": True,
                "op": "centroid",
                "centroid_xyz_m": np.round(region["centroid"], 6).tolist(),
                "n_points": region["n_points"],
                "frame_indices": (
                    [region["frame_index"]] if region["frame_index"] is not None else []
                ),
                "units": "metres",
            },
        )

    pts = region["points"]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = hi - lo
    return JSONResponse(
        status_code=200,
        content={
            "found": True,
            "op": "bbox3d",
            "min_xyz_m": np.round(lo, 6).tolist(),
            "max_xyz_m": np.round(hi, 6).tolist(),
            "extent_xyz_m": np.round(extent, 6).tolist(),
            "center_xyz_m": np.round((lo + hi) / 2.0, 6).tolist(),
            "volume_m3": round(float(np.prod(extent)), 8),
            "n_points": region["n_points"],
            "frame_indices": (
                [region["frame_index"]] if region["frame_index"] is not None else []
            ),
            "units": "metres",
        },
    )


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Turbo-ish false colour for a single depth frame, near = warm."""
    finite = np.isfinite(depth)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not finite.any():
        return out
    values = depth[finite]
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    # Simple 3-stop ramp: red (near) -> green -> blue (far).
    red = np.clip(1.0 - 2.0 * norm, 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * norm - 1.0)
    blue = np.clip(2.0 * norm - 1.0, 0.0, 1.0)
    for channel, ramp in enumerate((red, green, blue)):
        out[..., channel] = (ramp * 255).astype(np.uint8)
    out[~finite] = 0
    return out


@app.post("/render")
async def render(request: Request) -> JSONResponse:
    """Write a PNG so scaffolds that can read images can inspect results.

    Body: {"kind":"depth"|"mask"|"overlay", "array":"/work/...npy",
           "frame_index":0, "instance":0, "image":"/work/f0.png"}
    """
    payload = await read_json_body(request)
    alias = caller_alias(payload)
    work_base = resolve_work_base(payload)
    kind = payload.get("kind")
    if kind not in {"depth", "mask", "overlay"}:
        raise BadRequest("kind must be one of: depth, mask, overlay", got=repr(kind))

    raw_array = payload.get("array")
    if raw_array is None and isinstance(payload.get("arrays"), list):
        arrays = payload["arrays"]
        if len(arrays) != 1:
            raise BadRequest("pass exactly one array (or use the 'array' field)")
        raw_array = arrays[0]
    env = load_envelope(raw_array, "array", work_base=work_base)
    array = env["array"]
    frames = [int(i) for i in env["frame_indices"]]

    try:
        instance = int(payload.get("instance", 0))
    except (TypeError, ValueError) as exc:
        raise BadRequest("instance must be an integer") from exc

    if kind == "depth":
        if array.ndim != 3:
            raise BadRequest(
                "depth rendering needs a (N, H, W) array", shape=list(array.shape)
            )
        frame_index = int(payload.get("frame_index", frames[0] if frames else 0))
        if frame_index not in frames:
            raise BadRequest(
                f"frame {frame_index} not in array", available_frames=frames
            )
        rgb = _colorize_depth(array[frames.index(frame_index)].astype(np.float64))
        rendered_frames = [frame_index]
    elif kind == "mask":
        if array.ndim != 3:
            raise BadRequest(
                "mask rendering needs an (n_instances, H, W) array",
                shape=list(array.shape),
            )
        if instance >= array.shape[0]:
            raise BadRequest(
                f"instance {instance} out of range", n_instances=int(array.shape[0])
            )
        mask = array[instance].astype(bool)
        rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
        rgb[mask] = (255, 255, 255)
        rendered_frames = frames
    else:  # overlay
        image_path = resolve_in_work_dir(
            payload.get("image"), must_exist=True, field="image", work_base=work_base
        )
        base = np.asarray(load_image(image_path), dtype=np.float32)
        if array.ndim != 3:
            raise BadRequest(
                "overlay needs an (n_instances, H, W) mask array",
                shape=list(array.shape),
            )
        if instance >= array.shape[0]:
            raise BadRequest(
                f"instance {instance} out of range", n_instances=int(array.shape[0])
            )
        mask = array[instance].astype(bool)
        if mask.shape != base.shape[:2]:
            raise BadRequest(
                "mask resolution does not match the image",
                mask_shape=list(mask.shape),
                image_shape=list(base.shape[:2]),
            )
        tint = np.array([255.0, 40.0, 40.0], dtype=np.float32)
        alpha = float(payload.get("alpha", 0.5))
        blended = base.copy()
        blended[mask] = (1.0 - alpha) * base[mask] + alpha * tint
        rgb = np.clip(blended, 0, 255).astype(np.uint8)
        rendered_frames = frames

    with artifact_writer(
        f"render_{kind}", ".png", subdir="renders", work_base=work_base
    ) as writer:
        out_path = writer.path
        with writer.open() as handle:
            # Explicit format: PIL infers it from the filename, which it never
            # sees when handed an already-open file object.
            Image.fromarray(rgb).save(handle, format="PNG")
    return JSONResponse(
        status_code=200,
        content={
            "found": True,
            "kind": kind,
            "path": to_caller_path(str(out_path), alias, work_base=work_base),
            "frame_indices": rendered_frames,
            "size": {"height": int(rgb.shape[0]), "width": int(rgb.shape[1])},
            "source_array": to_caller_path(
                env["resolved_path"], alias, work_base=work_base
            ),
        },
    )


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="SpatialClaw perception REST gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    log_event(
        {
            "event": "startup",
            "host": args.host,
            "port": args.port,
            "work_dir": str(WORK_DIR),
            "spatialclaw_root": SPATIALCLAW_ROOT,
        }
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
