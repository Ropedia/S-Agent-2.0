from pathlib import Path

from pydantic import BaseModel

from sagent.models.task.id import GitTaskId, LocalTaskId, PackageTaskId


class DownloadedDatasetItem(BaseModel):
    id: GitTaskId | LocalTaskId | PackageTaskId
    downloaded_path: Path
