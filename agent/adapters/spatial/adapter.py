"""ERQA -> SAgent adapter (scaffold-agnostic spatial reasoning tasks).

ERQA (Embodied Reasoning Question Answer, Google DeepMind) is 400 single- and
multi-image multiple-choice questions about 3D layout, object state, camera
viewpoint change, and robot trajectories.

Each ERQA row becomes one SAgent task directory:

    <question_id>/
        instruction.md                    question + image list + perception API
        task.toml                         metadata, timeouts, network policy
        environment/
            Dockerfile                    CPU-only python:3.11-slim + curl/jq
            workspace/
                frame_0.png ...           decoded frames, deterministic order
                perception.py             gateway client, shared /perception mount
        tests/
            test.sh                       verifier, writes reward.txt
            score_erqa.py                 official scoring + extraction ladder
            expected_answer.txt           gold letter
        solution/
            solve.sh                      oracle: writes the gold letter

The task is scaffold-agnostic: nothing in it names an agent, and the only
contract with the solver is "leave one letter in /app/answer.txt".
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

#: The three source trees this adapter renders a task from. They are kept apart
#: because they are edited by different people at different rates:
#:
#:   prompt/         the user message. Four flat .md files -- what we tune most
#:                   often, so it is one directory deep and nothing else.
#:   skills/         one <skill-name>/SKILL.md each. The nested layout is NOT a
#:                   style choice: the SDK dispatches on the *filename*
#:                   (skills/skill.py Skill.load), and only a file literally
#:                   named SKILL.md gets is_agentskills_format=True. Flattening
#:                   to skills/<name>.md would silently downgrade every skill to
#:                   a resident repo skill (PITFALLS 1.5).
#:   task_template/  what actually goes into the container image: Dockerfile,
#:                   task.toml, workspace helpers, verifier. Rarely touched.
ADAPTER_DIR = Path(__file__).parent
PROMPT_DIR = ADAPTER_DIR / "prompt"
SKILLS_SRC_DIR = ADAPTER_DIR / "skills"
TEMPLATE_DIR = ADAPTER_DIR / "task_template"

# Gateway-side directory that task frames are staged into. The gateway only
# reads paths under its own work dir, so the adapter copies each task's frames
# there in addition to baking them into the image.
# Override with $GATEWAY_WORK_DIR. Must be a path the gateway process can read,
# since it only serves files under its own work dir.
DEFAULT_GATEWAY_WORK_DIR = Path(
    os.environ.get("GATEWAY_WORK_DIR", Path(__file__).resolve().parents[2] / "work")
)
# Historically just "erqa", staged as work/erqa/<question_id>/ with no configuration
# dimension -- that was Bug 2 (see gateway_prefix): all four configurations of the same
# question wrote and read the same directory and could race on it. New
# generations stage under "trials" instead, alongside the per-configuration key
# gateway_prefix() builds below; the old work/erqa/ tree from already-run
# experiments is untouched (nothing here reads or writes that name anymore).
GATEWAY_STAGE_SUBDIR = "trials"

IMAGE_PLACEHOLDER = "<image>"

# The template Dockerfile declares the agent uid/gid as build ARGs so it stays
# portable; the adapter rewrites their *default* values with this host's real
# ids when generating a task. Rewriting defaults rather than passing
# --build-arg is deliberate and necessary: SAgent builds task images through
# docker-compose-build.yaml, which declares only `context` and no `args`, and
# nothing on the `sagent run` command line (--mounts-json included) reaches a
# build argument. A value that cannot be passed at build time has to be in the
# file. Verified against sagent/environments/docker/ -- no build_args anywhere
# in the source tree.
DOCKERFILE_UID_ARG = "APP_UID"
DOCKERFILE_GID_ARG = "APP_GID"


def host_uid() -> int:
    """uid the gateway process runs as on this host (os.getuid, or 0)."""
    return os.getuid() if hasattr(os, "getuid") else 0


def host_gid() -> int:
    """gid the gateway process runs as on this host (os.getgid, or 0)."""
    return os.getgid() if hasattr(os, "getgid") else 0


@dataclass
class ERQATask:
    """One ERQA sample, already decoded from the parquet row."""

    question_id: str
    question: str
    question_type: str
    answer: str
    visual_indices: list[int] = field(default_factory=list)
    image_bytes: list[bytes] = field(default_factory=list)
    #: What `/app/answer.txt` is supposed to hold. "letter" is the original
    #: multiple-choice contract (ERQA, MMSI, MindCube, and SPBench's MCA half).
    #: "numeric" is SPBench's NA half -- a count, a distance or a size in
    #: metres, scored by Mean Relative Accuracy rather than exact match. It is
    #: a per-TASK property, not a per-run one: one SPBench run mixes both, so
    #: the instruction wording, the scorer and the verifier are all selected
    #: from this field rather than from a flag on the adapter.
    answer_kind: str = "letter"
    #: Basename of each image in the task workspace, parallel to `image_bytes`.
    #: None -- every loader before MMSI-Video -- names them `frame_<i>.png`,
    #: the name the whole stack assumes. MMSI-Video sets `v<K>_frame_<i>.png`
    #: (the video number rides in the name; the index stays GLOBAL so
    #: `camera:f<i>` / `@f<i>` / `f<i>:label` stay unique across videos) and
    #: `ref_<n>.png` for reference images, which must NOT count as frames: the
    #: container CLI reconstructs every frame file it finds, and a reference
    #: image inside a DA3 batch corrupts the scene. Names are validated at
    #: generation time (see ERQAAdapter._image_names).
    image_names: list[str] | None = None
    #: Extra files written verbatim into environment/workspace/ (name -> text),
    #: e.g. MMSI-Video's frames.json timestamp sidecar. None writes nothing.
    workspace_files: dict[str, str] | None = None
    #: Whether `<image>` markers are interleaved into the question block
    #: (ERQA's visual_indices contract, and the "all images first" prefix when
    #: the indices are empty). MMSI-Video turns it off: its question header
    #: already names every file, and 64 markers in a row would be noise.
    interleave_images: bool = True

    def image_name(self, index: int) -> str:
        """Workspace basename of image `index` (frame_<i>.png unless named)."""
        if self.image_names is None:
            return f"frame_{index}.png"
        return self.image_names[index]

    @property
    def num_images(self) -> int:
        return len(self.image_bytes)

    @property
    def is_multi_image(self) -> bool:
        return self.num_images > 1


# One block per vision mode. These differ ONLY
# in how the agent is told to obtain pixels -- question, options, gold answer
# and scorer are identical across modes, which is what keeps the configurations paired.
VISION_BLOCKS = {
    "disk": (
        "## Looking at the images\n\n"
        "The image files are on disk in this container at `/app/frame_0.png`, "
        "`/app/frame_1.png`, and so on. Your file editor's `view` command "
        "displays image files, so run `view` on a frame to look at it. Decide "
        "for yourself which frames are worth opening; you also have the "
        "perception API below for measurements."
    ),
    # Pixels only: no perception service at all. The instruction omits the whole
    # Perception API section and the task image ships no perception.py, so the
    # agent's only evidence channel is looking at the frames.
    #
    # This isolates what the disk configuration cannot: the disk configuration's 63.0% was produced
    # with perception available, and 31 of its 100 episodes used it (176 calls).
    # So "the agent opens frames itself" and "the agent can also measure" were
    # confounded in that configuration. This one separates them.
    "disk_only": (
        "## Looking at the images\n\n"
        "The image files are on disk in this container at `/app/frame_0.png`, "
        "`/app/frame_1.png`, and so on. Your file editor's `view` command "
        "displays image files, so run `view` on a frame to look at it. Decide "
        "for yourself which frames are worth opening. Looking at the frames is "
        "the only evidence available to you in this run."
    ),
}

#: The modes a task can be generated in.
RUNNABLE_VISION_MODES = tuple(k for k, v in VISION_BLOCKS.items() if v is not None)

#: Modes that get the Perception API section and a perception.py helper.
#: `disk_only` is deliberately absent: it ships neither, so the agent has no
#: perception channel to use even if it tried.
PERCEPTION_MODES = ("disk",)

#: root would never reach the container (same trap as vision_manifest.json).
SKILLS_DIR_NAME = "skills"


#: Marker the task image carries for non-letter tasks; read by the container
#: runtime's nudge/salvage layers AND by the fractional-reward scorers. Absent =
#: the original letter contract. Lives in environment/workspace/, which the
#: Dockerfile COPYs to /app, so both read it at /app/.answer_format.
ANSWER_FORMAT_FILE = ".answer_format"

#: The container-side helpers every perception configuration ships, in copy order. Each
#: comes from `workspace_dir` when a profile provides its own copy there, and
#: from the task template otherwise (see ERQAAdapter._workspace_source).
WORKSPACE_HELPERS = ("perception.py", "geom.py", "maskutil.py")
#: Names an ERQATask may give its images or extra workspace files: a bare
#: basename, no path separators, nothing hidden. Anything else is refused at
#: generation time rather than written somewhere the Dockerfile COPY would
#: not pick up or the CLI would misread.
_SAFE_WORKSPACE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

#: Network mode for the trial container, from $SAGENT_TASK_NETWORK_MODE.
#: Unset (the default) writes nothing and SAgent's own default applies -- every
#: existing task directory regenerates byte for byte.
#:
#: Set it to "host" where the trial container has to reach services running on
#: the DOCKER HOST rather than on a per-trial bridge. On the Aliyun DSW box the
#: platform's docker proxy puts containers on a network that cannot reach the
#: DSW container's own IP at all (measured: `10.11.0.3:8077` times out from a
#: bridge container, works from `--network=host`), so the gateway and LiteLLM
#: are unreachable without it.
#:
#: Delivered as a per-task `environment/docker-compose.yaml`, which SAgent
#: appends LAST and lets override scalars (see
#: src/sagent/environments/docker/docker.py, _get_docker_compose_paths) -- the
#: same mechanism its own docker-compose-no-network.yaml uses for
#: `network_mode: none`. No SAgent change needed.
TASK_NETWORK_MODE = os.environ.get("SAGENT_TASK_NETWORK_MODE", "").strip()

#: Block the trial container's internet egress, from $SAGENT_TASK_NO_EGRESS=1.
#: Unset (the default) writes nothing -- same byte-for-byte guarantee as above.
#:
#: Why: the benchmark container had full internet, and gemini-3.8-flash in the
#: no-tools configuration used `terminal` to search the web for the question text
#: (DuckDuckGo / GitHub code search / HuggingFace datasets-server rows -- it
#: pulled BLINK rows with "Answer: (A)"), and pip-installed opencv/scipy when
#: stuck. 15-19% of no-tool trials on the seed42 sets did this (2026-09-12,
#: sagent PITFALLS 7.14). Nothing benchmark-specific was found, but a benchmark
#: run must not depend on what the agent can fetch.
#:
#: How: per-task compose override turning off IP masquerade on the trial's
#: network. Without NAT, packets to the outside leave with a 172.x source and
#: never come back (measured: google / 8.8.8.8 = no connect), while the host's
#: own services (LiteLLM, perception gateway) stay reachable because that
#: traffic never needs NAT (measured: both 200). No sudo, no daemon config,
#: no SAgent change; `network_mode: none` is not an option because the agent
#: runtime inside the container must reach LiteLLM. DNS still resolves (the
#: host's resolver answers), so an agent sees "network unreachable", not
#: "no such host". Not a firewall: a router that routes 172.16/12 back to this
#: host would defeat it -- on such a LAN use an iptables DOCKER-USER rule.
TASK_NO_EGRESS = os.environ.get("SAGENT_TASK_NO_EGRESS", "").strip() in {
    "1",
    "true",
    "yes",
}


def task_compose_override() -> str:
    """The per-task `environment/docker-compose.yaml`, or "" to write none.

    SAgent appends this file last, so it can override scalars (network_mode)
    and add top-level keys (networks). Both switches are process-wide env, so
    every configuration of a sweep gets the same file -- the pairing invariant holds.
    """
    if TASK_NETWORK_MODE and TASK_NO_EGRESS:
        raise ValueError(
            "SAGENT_TASK_NETWORK_MODE and SAGENT_TASK_NO_EGRESS are exclusive: "
            "masquerade is a bridge-network option, host mode has no bridge"
        )
    if TASK_NETWORK_MODE:
        return "services:\n  main:\n    network_mode: " + TASK_NETWORK_MODE + "\n"
    if TASK_NO_EGRESS:
        return (
            "networks:\n"
            "  default:\n"
            "    driver_opts:\n"
            '      com.docker.network.bridge.enable_ip_masquerade: "false"\n'
        )
    return ""


#: What each answer contract needs: the instruction template that states it, the
#: scorer that grades it, the verifier that turns that into a reward, and the
#: marker text (None = write no marker, which is what keeps every letter task
#: byte-identical to the pre-2026-08-29 tree).
#:
#: One table rather than three branches because these four choices MUST move
#: together: an instruction asking for a number graded by the letter scorer
#: scores 0 on a correct answer and reads as a model failure (PITFALLS 4.12),
#: and a numeric task verified by test.sh loses the fractional reward entirely.
#:
#: `numeric_last` is `numeric` with one difference, carried in the marker:
#: SPAR-Bench's official extractor reads the LAST number in a prediction on
#: multi-view NA tasks and the first everywhere else.
ANSWER_CONTRACTS: dict[str, tuple[str, str, str, str | None]] = {
    "letter": ("instruction.md", "score_erqa.py", "test.sh", None),
    "numeric": (
        "instruction_numeric.md",
        "score_numeric.py",
        "test_numeric.sh",
        "number",
    ),
    "numeric_last": (
        "instruction_numeric.md",
        "score_numeric.py",
        "test_numeric.sh",
        "number_last",
    ),
    "vci": ("instruction_vci.md", "score_vci.py", "test_vci.sh", "vci"),
}


class ERQAAdapter:
    """Generates SAgent task directories from ERQA samples."""

    NAME = "spatial"

    def __init__(
        self,
        output_dir: Path,
        template_dir: Path | None = None,
        gateway_work_dir: Path | None = None,
        stage_images: bool = True,
        vision_mode: str = "disk",
        app_uid: int | None = None,
        app_gid: int | None = None,
        prompt_dir: Path | None = None,
        skills_dir: Path | None = None,
        workspace_dir: Path | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir or TEMPLATE_DIR)
        # `prompt_dir` and `skills_dir` are separate knobs, and they follow
        # `template_dir` only when that override actually contains them. Tests
        # copy `task_template/` alone into a scratch dir and pass it as
        # template_dir; making skills_dir blindly follow would then point at a
        # directory with no skills and trip the perception-configuration guard, masking
        # whatever the test was actually asserting.
        _tpl = self.template_dir
        self.prompt_dir = Path(
            prompt_dir
            if prompt_dir is not None
            else (_tpl / "prompt" if (_tpl / "prompt").is_dir() else PROMPT_DIR)
        )
        self._skills_dir = Path(
            skills_dir
            if skills_dir is not None
            else (_tpl / "skills" if (_tpl / "skills").is_dir() else SKILLS_SRC_DIR)
        )
        # A profile's own copy of the container-side CLI. None -- every profile
        # before video-v1 -- ships the template's helpers, byte for byte as
        # before. Given, the directory must exist and hold at least one of the
        # helpers: a typo that silently shipped the template would make a
        # profile's CLI changes look like they changed nothing (the same
        # failure shape skills_dir guards against in the loaders).
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        if self.workspace_dir is not None:
            if not self.workspace_dir.is_dir():
                raise FileNotFoundError(
                    f"workspace_dir does not exist: {self.workspace_dir}"
                )
            if not any((self.workspace_dir / n).is_file() for n in WORKSPACE_HELPERS):
                raise FileNotFoundError(
                    f"workspace_dir {self.workspace_dir} holds none of "
                    f"{WORKSPACE_HELPERS}; nothing to override"
                )
        self.gateway_work_dir = Path(gateway_work_dir or DEFAULT_GATEWAY_WORK_DIR)
        self.stage_images = stage_images
        # Resolved at generation time, not baked into the template: the image
        # must run as whatever uid the gateway on THIS host runs as, and the
        # template has to stay portable. See _render_dockerfile.
        self.app_uid = int(app_uid) if app_uid is not None else host_uid()
        self.app_gid = int(app_gid) if app_gid is not None else host_gid()
        if vision_mode not in RUNNABLE_VISION_MODES:
            raise ValueError(
                f"vision_mode must be one of {sorted(RUNNABLE_VISION_MODES)}, "
                f"got {vision_mode!r}"
            )
        self.vision_mode = vision_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- skills ------------------------------------------------------------

    @property
    def skills_dir(self) -> Path:
        """Source directory holding one subdirectory per skill.

        Sits beside the adapter rather than under the task template, because
        skills are prompt material we edit constantly, not container build
        input. Overridable through the constructor for tests.
        """
        return self._skills_dir

    def skill_files(self) -> list[Path]:
        """Every `<skill>/SKILL.md` under the template, sorted by skill name.

        Sorted so both delivery modes see a deterministic order: the inline
        concatenation must be byte-stable across regeneration, or task
        directories stop being reproducible.
        """
        if not self.skills_dir.is_dir():
            return []
        return sorted(
            (d / "SKILL.md" for d in self.skills_dir.iterdir() if d.is_dir()),
            key=lambda p: p.parent.name,
        )

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Return (frontmatter fields, body) for one SKILL.md.

        Deliberately a tiny hand-rolled parser rather than a YAML dependency:
        the frontmatter here is a fixed two-key block, and the adapter runs on
        the host where PyYAML is not guaranteed.
        """
        if not text.startswith("---\n"):
            raise ValueError("SKILL.md must open with a '---' frontmatter fence")
        end = text.index("\n---\n", 3)
        fields: dict[str, str] = {}
        for line in text[4:end].splitlines():
            if not line.strip():
                continue
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        return fields, text[end + 5 :].lstrip("\n")

    def skill_bodies(self) -> str:
        """Skill bodies concatenated for the `inline` configuration.

        Frontmatter is stripped -- `name`/`description` exist to let an agent
        decide whether to invoke a skill, and in this mode there is nothing to
        invoke. The bodies themselves, which are what the agent actually acts on,
        are the same bytes the `skill` configuration ships.
        """
        parts = []
        for path in self.skill_files():
            _fields, body = self._split_frontmatter(path.read_text())
            parts.append(body.rstrip("\n"))
        return "\n\n".join(parts)

    def skill_index(self) -> str:
        """Names and trigger descriptions, in the same order as the bodies."""
        lines: list[str] = []
        for path in self.skill_files():
            fields, _body = self._split_frontmatter(path.read_text())
            name = fields.get("name", path.parent.name)
            description = fields.get("description", "").strip()
            lines.append(f"- **{name}** — {description}")
        return "\n".join(lines)

    def skill_inline_text(self, include_index: bool = False) -> str:
        """Return resident skill text, optionally with its routing index."""
        body = self.skill_bodies()
        if not include_index:
            return body
        index = self.skill_index()
        if not index:
            return body
        return f"### Available skills\n\n{index}\n\n### Skill details\n\n{body}"

    # -- rendering helpers -------------------------------------------------

    @staticmethod
    def _example_letter(task: "ERQATask", letters: list[str]) -> str:
        """Pick the letter shown in the worked `echo -n "X"` example.

        Must carry **no information about the gold answer**. Chosen by hashing the
        question id, which is stable across regeneration (so a task directory is
        reproducible) but uncorrelated with the answer.

        Two-choice items are special-cased to a fixed literal: with only A and B,
        *any* rule that avoids the gold letter reveals it outright.
        """
        if len(letters) <= 2:
            return letters[0]
        digest = hashlib.sha256(str(task.question_id).encode()).hexdigest()
        return letters[int(digest, 16) % len(letters)]

    @staticmethod
    def _render(template_text: str, context: dict[str, Any]) -> str:
        """Render with str.format so templates use {name} and literal {{ }}."""
        return template_text.format(**context)

    def _render_dockerfile(self, template_text: str) -> str:
        """Bake this host's uid/gid into the ARG defaults of the Dockerfile.

        Not ``_render``/``str.format``: a Dockerfile is full of shell ``${VAR}``
        expansions that ``str.format`` would either mangle or choke on, and
        escaping every one of them would make the template unreadable and
        fragile. A line-anchored rewrite of just the two ``ARG`` defaults keeps
        the template a valid, buildable, host-independent Dockerfile.

        The rewrite is required to be a no-op-proof edit: if either ARG line
        ever disappears from the template, this raises rather than silently
        emitting an image that runs as the wrong uid -- a silent fallback here
        would resurrect Bug A with no visible symptom until a 100-task run came
        back with 484 bad_requests.
        """
        rendered = template_text
        for arg, value in (
            (DOCKERFILE_UID_ARG, self.app_uid),
            (DOCKERFILE_GID_ARG, self.app_gid),
        ):
            pattern = re.compile(rf"^ARG {re.escape(arg)}=.*$", re.MULTILINE)
            rendered, count = pattern.subn(f"ARG {arg}={value}", rendered)
            if count != 1:
                raise ValueError(
                    f"expected exactly one 'ARG {arg}=' line in the template "
                    f"Dockerfile, found {count}"
                )
        return rendered

    @staticmethod
    def interleave_question(
        question: str, visual_indices: list[int], num_images: int
    ) -> str:
        """Insert <image> markers at ERQA's character offsets.

        Mirrors the reference loader (spatial_agent/evals/erqa.py
        _interleave_question), which in turn mirrors ERQA's own eval_harness:
        only the first ``num_images`` indices are used, because the official
        code does ``zip(pil_images, visual_indices)``.

        Empty or all-zero indices mean "all images come before the question",
        which is the case for 183 of the 400 rows (visual_indices is simply
        absent there) plus every single-image row whose index is 0.
        """
        if num_images == 0:
            return question

        used = list(visual_indices[:num_images])

        if not used or all(idx == 0 for idx in used):
            prefix = " ".join(IMAGE_PLACEHOLDER for _ in range(num_images))
            return f"{prefix}\n{question}"

        pairs = sorted(enumerate(used), key=lambda pair: pair[1])

        parts: list[str] = []
        last_pos = 0
        for _, char_idx in pairs:
            if char_idx == 0:
                parts.append(IMAGE_PLACEHOLDER)
                continue
            clamped = min(char_idx, len(question))
            if clamped > last_pos:
                parts.append(question[last_pos:clamped])
            parts.append(IMAGE_PLACEHOLDER)
            last_pos = clamped

        if last_pos < len(question):
            parts.append(question[last_pos:])

        result = "".join(parts)
        if not result.strip():
            prefix = " ".join(IMAGE_PLACEHOLDER for _ in range(num_images))
            return f"{prefix}\n{question}"
        return result

    @staticmethod
    def _numbered_question(interleaved: str, names: int | list[str]) -> str:
        """Replace each <image> marker with a concrete frame reference.

        The agent sees "[image: frame_0.png]" exactly where ERQA puts an image,
        so positional wording in the question ("the previous image", "frame 1",
        "the following images") still resolves. `names` is the per-image
        basename list; an int (the old signature) means the default names.
        """
        if isinstance(names, int):
            names = [f"frame_{i}.png" for i in range(names)]
        out = interleaved
        for name in names:
            out = out.replace(IMAGE_PLACEHOLDER, f"[image: {name}]", 1)
        return out

    @staticmethod
    def _image_names(task: ERQATask) -> list[str]:
        """Validated per-image basenames for `task` (see ERQATask.image_names)."""
        if task.image_names is None:
            return [f"frame_{i}.png" for i in range(task.num_images)]
        names = list(task.image_names)
        if len(names) != task.num_images:
            raise ValueError(
                f"{task.question_id}: image_names has {len(names)} entries for "
                f"{task.num_images} images"
            )
        for name in names:
            if not _SAFE_WORKSPACE_NAME_RE.fullmatch(name) or not name.endswith(".png"):
                raise ValueError(
                    f"{task.question_id}: image name {name!r} must be a bare "
                    "basename ending in .png"
                )
            if name in WORKSPACE_HELPERS or name in (
                ANSWER_FORMAT_FILE,
                SKILLS_DIR_NAME,
            ):
                raise ValueError(
                    f"{task.question_id}: image name {name!r} collides with a "
                    "workspace helper"
                )
        if len(set(names)) != len(names):
            raise ValueError(f"{task.question_id}: image names must be unique")
        return names

    def _workspace_source(self, name: str) -> Path:
        """Where helper `name` is copied from: the profile's workspace_dir when
        it ships that file, otherwise the task template."""
        if self.workspace_dir is not None and (self.workspace_dir / name).is_file():
            return self.workspace_dir / name
        return self.template_dir / "environment" / "workspace" / name

    def _vision_block(self, task: ERQATask, names: list[str]) -> str:
        """The configuration's pixel-delivery paragraph, with real file names in it.

        The disk configurations name the first two files as examples. With the default
        names the text is returned untouched (byte-identical instruction.md
        for every existing task); with loader-supplied names the examples are
        rewritten to files that exist, so an MMSI-Video agent is not told to
        open `/app/frame_0.png` when the file is `/app/v1_frame_0.png`.
        """
        block = VISION_BLOCKS[self.vision_mode]
        if task.image_names is None or not names:
            return block
        if len(names) >= 2:
            return block.replace("/app/frame_0.png", f"/app/{names[0]}").replace(
                "/app/frame_1.png", f"/app/{names[1]}"
            )
        return block.replace(
            "`/app/frame_0.png`, `/app/frame_1.png`, and so on", f"`/app/{names[0]}`"
        )

    def _image_block(self, task: ERQATask) -> str:
        if task.num_images == 0:
            return "_This question has no images._"

        lines = [
            f"This question has {task.num_images} "
            f"image{'s' if task.num_images != 1 else ''}, in order:",
            "",
        ]
        # Configurations without perception must not be told a perception channel
        # exists; naming a service the container cannot reach would be a dead
        # reference and would leak the existence of a channel this configuration is
        # defined to lack.
        #
        # 2026-08-21: the perception configurations used to add "copy it into /perception/
        # ... to make it readable by the perception gateway too". That is wrong
        # twice over. It is factually stale -- perception.py stages the frames
        # itself (ensure_frames_staged), and has since the 100-question run --
        # and it contradicts the Perception API section, which tells the agent
        # the helper "takes no setup". Worse, its docstring records exactly why
        # it moved out of the instruction: putting `cp /app/frame_*.png
        # /perception/` in front of the agent HALVED perception adoption
        # (31 -> 14 episodes). So both configurations now get the same plain line, and
        # this loop no longer branches at all.
        for i in range(task.num_images):
            name = task.image_name(i)
            lines.append(f"{i + 1}. `{name}` — in this container at `/app/{name}`")
        if task.num_images > 1:
            # Without interleaving (MMSI-Video) there are no markers to point
            # at: the question's own header names the files instead.
            markers = (
                ", and the `[image: frame_N.png]` markers in the question show "
                "where each one belongs in the text."
                if task.interleave_images
                else "."
            )
            lines += [
                "",
                "The numbering above is the dataset's own image order" + markers,
            ]
        return "\n".join(lines)

    @staticmethod
    def _choice_letters(question: str) -> list[str]:
        """Letters actually offered by this question, in order.

        Nearly every ERQA question offers A-D, but the letters are detected from
        the text rather than assumed so the instruction never advertises an
        option that does not exist.
        """
        # Options must appear in ascending positional order, not merely be present.
        #
        # Scanning each letter independently picks up incidental matches from the
        # *choice texts*. ERQA_306 asks which musical note a key plays --
        # "A. F-sharp. B. E. C. F. D. D" -- so "E." and "F." look like options and
        # the instruction advertised "A, B, C, D, E, or F" for a 4-option question,
        # with a decoy letter that is not a valid choice at all. A contiguity check
        # alone does not help, because the stray letters here happen to be exactly
        # E and F and so continue the sequence.
        #
        # Real option labels increase monotonically through the string, whereas the
        # strays sit earlier (inside "B. E." and "C. F."). Requiring each label to
        # appear after the previous one rejects them.
        found: list[str] = []
        last_pos = -1
        for letter in "ABCDEFGH":
            positions = [
                question.find(pattern)
                for pattern in (f"{letter}.", f"{letter})", f"({letter})")
                if question.find(pattern) > last_pos
            ]
            if not positions:
                break
            last_pos = min(positions)
            found.append(letter)
        return found or ["A", "B", "C", "D"]

    # -- image decoding ----------------------------------------------------

    @staticmethod
    def _decode_to_png(raw: bytes, destination: Path) -> None:
        """Decode embedded image bytes and write a real PNG.

        ERQA stores JPEG bytes in the parquet. Re-encoding to PNG makes the
        on-disk artifact lossless and self-describing, and guarantees the file
        extension matches the content, which matters because the agent and the
        gateway both open these by name.
        """
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as handle:
            handle.load()
            rgb = handle.convert("RGB")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination, format="PNG")

    def gateway_prefix(self, question_id: str) -> str:
        """Gateway-side directory for this task's frames.

        Keyed by (vision_mode, question_id), not question_id alone.

        Bug 2: the old key was question_id only, so every configuration shared one
        directory (work/erqa/<question_id>/frame_N.png). Nothing stopped one
        configuration's task generation (a `shutil.copy2` here) from overwriting the
        very file another configuration's already-running trial was asking the gateway
        to read mid-request. Content happened to be identical -- frames are
        decoded from the same fixed parquet row regardless of which configuration
        writes them, verified by sha256 -- so the collisions that already
        happened were harmless by luck (an idempotent overwrite), not by
        design (a reader can still catch a torn/partial file mid-`copy2`).
        Adding the vision_mode segment gives each configuration its own directory, so
        two configurations of the same question_id can no longer race on the same path.

        This key is generation-time, not run-time. SAgent does not assign a
        trial name (e.g. "ERQA_1__trbao5g") until a task is actually
        scheduled, and this method runs earlier, while building the task
        directory -- so a per-trial key is not available here. vision_mode is
        the finest dimension this method can see.

        Residual limitation: two runs of the *same* configuration on the same
        question_id (e.g. a re-`--overwrite`, or two concurrent trials of one
        configuration both hitting the gateway for this question) still share this
        directory. Accepted for the reason above -- same parquet row, same
        bytes, last-writer-wins is idempotent -- and left as a known gap
        rather than solved here because closing it needs the per-trial mount
        that task 1c wires up at run time (adapter.py only runs at generation
        time and cannot see a trial name).
        """
        return str(
            self.gateway_work_dir
            / GATEWAY_STAGE_SUBDIR
            / self.vision_mode
            / question_id
        )

    # -- generation --------------------------------------------------------

    def generate_task(self, task: ERQATask, overwrite: bool = False) -> Path | None:
        """Write one SAgent task directory. Returns None if it already exists."""
        task_dir = self.output_dir / task.question_id

        if task_dir.exists():
            if not overwrite:
                return None
            shutil.rmtree(task_dir)

        workspace = task_dir / "environment" / "workspace"
        workspace.mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "solution").mkdir()

        image_prefix = self.gateway_prefix(task.question_id)
        names = self._image_names(task)

        # 1. Frames, in ERQA's own order, as PNG.
        for idx, raw in enumerate(task.image_bytes):
            self._decode_to_png(raw, workspace / names[idx])

        # 2. Same frames staged where the gateway can read them. Skipped for
        #    configurations outside PERCEPTION_MODES (currently just disk_only): those
        #    configurations ship no perception.py and no gateway address ever reaches
        #    the container (see the instruction-block gating below), so the
        #    gateway will never be asked to read a staged frame for this
        #    question. Staging one anyway was pure waste -- 400 disk_only
        #    tasks warmed 400 directories nothing ever read.
        if (
            self.stage_images
            and task.image_bytes
            and self.vision_mode in PERCEPTION_MODES
        ):
            stage_dir = Path(image_prefix)
            stage_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(task.num_images):
                shutil.copy2(workspace / names[idx], stage_dir / names[idx])

        # 3. Perception helper, copied verbatim (no per-task substitution --
        #    it discovers its own gateway work_dir at runtime from
        #    /proc/self/mountinfo, see PERCEPTION_WORK_DIR in the template).
        #    Omitted entirely for configurations outside PERCEPTION_MODES, so the
        #    container carries no way to reach the gateway.
        #    Each helper comes from the profile's workspace_dir when it has
        #    its own copy (video-v1's perception.py), else from the template.
        if self.vision_mode in PERCEPTION_MODES:
            for helper in WORKSPACE_HELPERS:
                shutil.copy2(self._workspace_source(helper), workspace / helper)

        # 3a. Answer-format marker, numeric tasks only.
        #
        # The container-side runtime (scripts/runtime_patch/
        # run_openhands_patched.py) has two repair layers for a missing
        # answer file -- the nudge that asks a model that forgot, and the
        # salvage that recovers a submission the wire ate -- and both were
        # written when every task wanted a letter. On a numeric task the nudge
        # asked for "a single option letter" and the salvage looked for one, so
        # an SPBench NA trial whose write never landed could not be rescued at
        # all while its multiple-choice neighbour could. That is a systematic
        # penalty on 66% of SPBench, not a model result.
        #
        # A FILE rather than an env var because the answer kind is per TASK and
        # the runtime env is per RUN: one SPBench job holds both kinds. Written
        # only for numeric tasks, so every existing task directory regenerates
        # byte for byte and "absent means letter" stays the default.
        #
        # It says nothing about the gold answer -- only which shape the answer
        # file takes, which instruction.md states in plain text anyway.
        try:
            contract = ANSWER_CONTRACTS[task.answer_kind]
        except KeyError:
            raise ValueError(
                f"{task.question_id}: unknown answer_kind {task.answer_kind!r}; "
                f"expected one of {sorted(ANSWER_CONTRACTS)}"
            ) from None
        instruction_name, scorer_name, verifier_name, marker = contract
        if marker is not None:
            (workspace / ANSWER_FORMAT_FILE).write_text(f"{marker}\n")

        # 3a'. Loader-supplied sidecars (MMSI-Video's frames.json). Written
        #      into the workspace so the Dockerfile's `COPY workspace/ /app/`
        #      carries them; names are checked against everything else that
        #      lands there so a sidecar can never shadow a helper or a frame.
        for name, text in sorted((task.workspace_files or {}).items()):
            if not _SAFE_WORKSPACE_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"{task.question_id}: workspace file name {name!r} must be a "
                    "bare basename (letters, digits, '_', '-', '.')"
                )
            if (
                name in WORKSPACE_HELPERS
                or name in names
                or name in (ANSWER_FORMAT_FILE, "vision_manifest.json", SKILLS_DIR_NAME)
            ):
                raise ValueError(
                    f"{task.question_id}: workspace file {name!r} collides with a "
                    "file the adapter writes itself"
                )
            (workspace / name).write_text(text)

        compose = task_compose_override()
        if compose:
            (task_dir / "environment" / "docker-compose.yaml").write_text(compose)

        # 3b. Skills. Every perception configuration needs them present REGARDLESS of
        #     delivery mode -- the check below is deliberately outside the
        #
        #     Why it has to be here and not only in the skill branch: `inline`
        #     consumes the same files through skill_bodies(), which joins an
        #     empty list into "" and returns it. Generation then completed
        #     normally and a perception configuration shipped an instruction with NO
        #     perception documentation, reporting success either way. That is
        #     a run that would
        #     have measured "no docs at all" while the report said "docs
        #     resident in every instruction" -- and no artifact distinguishes
        #     the two after the fact.
        #
        #     Same failure class as the 2026-08-20 max_tokens truncation: the
        #     pipeline is connected, every existing check passes, and the only
        #     symptom is a plausible-looking number (docs/PITFALLS.md 4.1d).
        #     Fail at generation time, where the cause is still visible.
        #
        #     disk_only is exempt by construction, not by accident: it is not a
        #     PERCEPTION_MODE, ships no perception facilities, and with no
        #     SKILL.md the SDK mounts no InvokeSkillTool at all
        #     (agent/base.py:548-566). "Registered but never triggered" would
        #     not do -- description osmosis (arXiv 2607.22520) shows an unused
        #     skill still shifts behaviour, the same shape as the blind configuration's
        #     13% contamination.
        if self.vision_mode in PERCEPTION_MODES:
            skill_files = self.skill_files()
            if not skill_files:
                raise ValueError(
                    f"vision_mode={self.vision_mode!r} is a perception configuration and "
                    f"needs at least one SKILL.md under {self.skills_dir}, "
                    f"found none (the instruction would otherwise carry no "
                    f"perception documentation)"
                )
        else:
            skill_files = []

        # Only `skill` mode stages them into the workspace. Under `inline` the
        # same bytes are already concatenated into instruction.md, so shipping
        # SKILL.md too would register the skills AND keep them resident, which
        # is not what the profile defines. The Dockerfile's recursive
        # `COPY workspace/ /app/` puts what we do write at /app/skills
        # no Dockerfile change is needed.
        (task_dir / "environment" / "Dockerfile").write_text(
            self._render_dockerfile(
                (self.template_dir / "environment" / "Dockerfile").read_text()
            )
        )

        # 5. instruction.md
        interleaved = (
            self.interleave_question(
                task.question, task.visual_indices, task.num_images
            )
            if task.interleave_images
            else task.question
        )
        letters = self._choice_letters(task.question)
        skill_bodies = self.skill_bodies()
        skill_index = ""

        # The Perception API section is a separate fragment so a configuration can omit it
        # without leaving a dangling reference to a service it cannot reach.
        if self.vision_mode in PERCEPTION_MODES:
            perception_block = (self.prompt_dir / "perception_api.md").read_text()
            perception_block = perception_block.replace("{skill_index}", skill_index)
            perception_block = perception_block.replace("{skill_bodies}", skill_bodies)
        else:
            perception_block = ""

        # The step-by-step working procedure. Split by configuration for the same reason
        # perception_block is: a configuration without a perception channel must not be
        # told to reach for one. tests/test_skill_delivery.py asserts the word
        # never appears in disk_only's instruction, and it caught exactly this
        # -- an unconditional workflow section leaked the channel's existence to
        # the configuration that is defined by lacking it.
        #
        # Inlined before rendering, like perception_block, so any braces inside
        # survive str.format untouched.
        workflow_name = (
            "workflow.md"
            if self.vision_mode in PERCEPTION_MODES
            else "workflow_nopercept.md"
        )
        workflow_block = (self.prompt_dir / workflow_name).read_text()

        # Numeric tasks get their own top-level template. A separate FILE (not a
        # branch inside instruction.md) keeps every existing prompt profile
        # byte-identical: seg-align-v19's four fragments are pinned as
        # byte-identical to v18, and a letter task must still render exactly the
        # instruction.md it rendered before this field existed.
        instruction_path = self.prompt_dir / instruction_name
        if not instruction_path.is_file():
            # Fail loudly. Silently falling back to instruction.md would tell the
            # agent to write a letter for a question that has no options, and
            # every such task would score 0 while looking like a model failure.
            raise FileNotFoundError(
                f"{instruction_path} not found; an answer_kind={task.answer_kind!r} "
                f"task needs {instruction_name} in {self.prompt_dir}"
            )
        instruction_template = instruction_path.read_text()
        perception_sentinel = "__BASELINE_PERCEPTION_BLOCK__"
        workflow_sentinel = "__BASELINE_WORKFLOW_BLOCK__"
        instruction_template = instruction_template.replace(
            "{perception_block}", perception_sentinel
        )
        instruction_template = instruction_template.replace(
            "{workflow_block}", workflow_sentinel
        )
        instruction = self._render(
            instruction_template,
            {
                "question_block": self._numbered_question(interleaved, names),
                "image_block": self._image_block(task),
                "vision_block": self._vision_block(task, names),
                "skill_bodies": skill_bodies,
                "skill_index": skill_index,
                "choice_letters": ", ".join(letters[:-1]) + f", or {letters[-1]}"
                if len(letters) > 1
                else letters[0],
                # ANSWER-LEAK FIX. The previous version picked the first letter
                # that is *not* the gold answer. `next()` over an ordered list is
                # deterministic, so the rule inverted perfectly: example "B" meant
                # gold "A" every time, and example "A" eliminated one distractor.
                # Measured on the first 100 tasks: 31/100 golds fully determined by
                # instruction.md alone, and a blind solver exploiting the pattern
                # scored 0.407 against 0.278 chance. That invalidated a running
                # 100-instance experiment.
                #
                # The decoy is now chosen by hashing the question id, so it carries
                # no information about the gold answer. Two-choice items get a fixed
                # literal, because there "not the gold" is unavoidably revealing.
                # Falls back to the first letter only if the question
                # somehow offers a single option.
                "example_letter": self._example_letter(task, letters),
            },
        )
        instruction = instruction.replace(perception_sentinel, perception_block)
        instruction = instruction.replace(workflow_sentinel, workflow_block)
        (task_dir / "instruction.md").write_text(instruction)

        # 6. task.toml
        task_toml = self._render(
            (self.template_dir / "task.toml").read_text(),
            {
                "difficulty": "hard" if task.is_multi_image else "medium",
                "question_id": task.question_id,
                "question_type": task.question_type,
                "question_type_tag": task.question_type.lower().replace(" ", "-"),
                "image_count_tag": "multi-image"
                if task.is_multi_image
                else "single-image",
                "num_images": task.num_images,
                "visual_indices": ",".join(str(i) for i in task.visual_indices),
                # SAgent runs the agent and verifier via `docker exec -u <this>`.
                # Must equal the image's APP_UID or /app/answer.txt is
                # unwritable and every task scores 0.
                "app_uid": self.app_uid,
            },
        )
        (task_dir / "task.toml").write_text(task_toml)

        # 7. Verifier: test.sh + scorer + gold answer.
        #
        # Both scorers land in tests/ under their own names, but the verifier is
        # always tests/test.sh -- task.toml names no scorer, SAgent just runs
        # test.sh. The numeric one writes a FRACTIONAL reward (MRA), which is
        # why it is a different test.sh and not a flag: the letter verifier's
        # contract is "exit code 0/1 -> reward 1/0", and MRA has no exit code.
        shutil.copy2(
            self.template_dir / "tests" / verifier_name, task_dir / "tests" / "test.sh"
        )
        shutil.copy2(
            self.template_dir / "tests" / scorer_name, task_dir / "tests" / scorer_name
        )
        (task_dir / "tests" / "test.sh").chmod(0o755)
        (task_dir / "tests" / scorer_name).chmod(0o755)
        (task_dir / "tests" / "expected_answer.txt").write_text(task.answer.strip())

        # 8. Oracle.
        solve_path = task_dir / "solution" / "solve.sh"
        solve_path.write_text(
            (self.template_dir / "solution" / "solve.sh")
            .read_text()
            .replace("{answer}", task.answer.strip())
        )
        solve_path.chmod(0o755)

        return task_dir

    def generate_tasks(
        self, tasks: list[ERQATask], overwrite: bool = False, workers: int = 1
    ) -> tuple[list[Path], list[str]]:
        """Generate many task directories, preserving input order.

        `workers` > 1 runs generate_task in a process pool. Tasks are
        independent -- each writes only its own task_dir and its own
        gateway_prefix, and generate_task reads no process-global state -- so
        the output is byte-identical to the serial path; the tests diff the two
        trees. Why bother: the 1106-question video set is ~40k PNG encodes on
        one core (~100 minutes) while every GPU waits for it.
        """
        pending: list[tuple[int, ERQATask]] = []
        skipped_at: dict[int, str] = {}
        for i, task in enumerate(tasks):
            if not task.answer.strip():
                skipped_at[i] = f"{task.question_id} (empty gold answer)"
            elif not task.image_bytes:
                skipped_at[i] = f"{task.question_id} (no images)"
            else:
                pending.append((i, task))

        run = partial(self.generate_task, overwrite=overwrite)
        if workers > 1 and len(pending) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, [task for _, task in pending]))
        else:
            results = [run(task) for _, task in pending]

        success: list[Path] = []
        skipped: list[str] = []
        by_index = {i: result for (i, _), result in zip(pending, results)}
        for i, task in enumerate(tasks):
            if i in skipped_at:
                skipped.append(skipped_at[i])
            elif by_index[i] is None:
                skipped.append(f"{task.question_id} (already exists)")
            else:
                success.append(by_index[i])
        return success, skipped
