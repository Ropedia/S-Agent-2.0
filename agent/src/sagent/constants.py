from pathlib import Path

CACHE_DIR = Path("~/.cache/sagent").expanduser()
TASK_CACHE_DIR = CACHE_DIR / "tasks"
PACKAGE_CACHE_DIR = CACHE_DIR / "tasks" / "packages"
DATASET_CACHE_DIR = CACHE_DIR / "datasets"
NOTIFICATIONS_PATH = CACHE_DIR / "notifications.json"
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/laude-institute/sagent/main/registry.json"
)
SAGENT_REGISTRY_WEBSITE_URL = "https://registry.sagentframework.com"
SAGENT_REGISTRY_TASKS_URL = f"{SAGENT_REGISTRY_WEBSITE_URL}/tasks"
SAGENT_REGISTRY_DATASETS_URL = f"{SAGENT_REGISTRY_WEBSITE_URL}/datasets"
ARCHIVE_FILENAME = "dist.tar.gz"
ORG_NAME_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
