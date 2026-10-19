import errno
import hashlib
import os
import re
import stat
from pathlib import Path

from sagent.models.task.config import TaskConfig
from sagent.models.task.paths import TaskPaths

# Matches canary lines: HTML comments (<!-- ...canary... -->) or hash comments (# ...canary...)
_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


def strip_canary(text: str) -> str:
    """Strip canary marker lines from the beginning of instruction text.

    Canary strings are data-provenance markers embedded in benchmark files to prevent
    training-data contamination. They should not be shown to agents. This function
    removes leading lines that match common canary comment formats (HTML comments or
    hash comments containing the word "canary"), plus any blank lines that follow.
    """
    lines = text.split("\n")
    idx = 0
    # Remove leading canary comment lines
    while idx < len(lines) and _CANARY_LINE_RE.match(lines[idx].strip()):
        idx += 1
    # Remove any blank lines immediately after the canary block
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return "\n".join(lines[idx:])


class Task:
    """
    Represents a task with the following directory structure:

    ├── instruction.md
    ├── task.toml
    ├── environment/
    │   ├── [docker-compose.yaml | Dockerfile | singularity-compose.yaml | etc.]
    │   └── ...
    ├── solution/         # copied to container @ /solution by OracleAgent
    │   ├── solve.sh
    │   └── ...
    └── tests/            # copied to container @ /tests by Evaluator
        ├── test.sh
        └── ...
    """

    def __init__(self, task_dir: Path | str):
        """
        Initialize a Task from a directory path.

        Args:
            task_dir: Path to the task directory
        """
        self._task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self._task_dir)
        self.instruction = strip_canary(self.paths.instruction_path.read_text())
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text())
        if self.config.task is not None:
            self.name = self.config.task.name
        else:
            self.name = self.paths.task_dir.name

    @property
    def checksum(self) -> str:
        """Generate a deterministic hash for the task based on its entire directory content."""
        from dirhash import dirhash

        try:
            return dirhash(self._task_dir, "sha256")
        except OSError as exc:
            if not _is_symlink_hash_error(exc):
                raise

            return _safe_task_dir_hash(self._task_dir)
        except Exception as exc:
            if type(exc).__name__ != "SymlinkRecursionError":
                raise

            return _safe_task_dir_hash(self._task_dir)

    @property
    def task_dir(self) -> Path:
        """Public accessor for the task directory."""
        return self._task_dir


def _safe_task_dir_hash(task_dir: Path) -> str:
    """Hash task contents without following symlinks.

    Some benchmark repos intentionally contain invalid or self-referential
    symlinks. Include symlink metadata in the hash, but never resolve targets.
    """
    outer = hashlib.sha256()
    for root, dirnames, filenames in os.walk(task_dir, followlinks=False):
        root_path = Path(root)
        dirnames.sort()
        filenames.sort()

        for name in list(dirnames):
            path = root_path / name
            rel = path.relative_to(task_dir).as_posix()
            if _is_link_like(path):
                outer.update(_link_hash_entry(path, rel))
                dirnames.remove(name)

        for name in filenames:
            path = root_path / name
            rel = path.relative_to(task_dir).as_posix()
            if _is_link_like(path):
                outer.update(_link_hash_entry(path, rel))
                continue

            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                outer.update(_unreadable_hash_entry(path, rel))
                continue

            outer.update(f"F\0{rel}\0{file_hash}\n".encode())

    return outer.hexdigest()


def _is_symlink_hash_error(exc: OSError) -> bool:
    return exc.errno == errno.ELOOP or getattr(exc, "winerror", None) == 1921


def _is_link_like(path: Path) -> bool:
    try:
        if os.path.islink(path):
            return True
        path_stat = path.lstat()
    except OSError:
        return False

    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(path_stat, "st_file_attributes", 0)
    return bool(reparse_point and file_attributes & reparse_point)


def _link_hash_entry(path: Path, rel: str) -> bytes:
    try:
        target = os.readlink(path)
    except OSError:
        path_stat = path.lstat()
        target = f"<unresolved:{path_stat.st_mode}:{path_stat.st_size}>"

    return f"L\0{rel}\0{target}\n".encode()


def _unreadable_hash_entry(path: Path, rel: str) -> bytes:
    path_stat = path.lstat()
    return f"U\0{rel}\0{path_stat.st_mode}\0{path_stat.st_size}\n".encode()
