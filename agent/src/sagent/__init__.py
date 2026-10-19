import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Core classes
    from sagent.agents.base import BaseAgent
    from sagent.environments.base import BaseEnvironment, ExecResult
    from sagent.job import Job

    # Agent models
    from sagent.models.agent.context import AgentContext
    from sagent.models.agent.name import AgentName
    from sagent.models.dataset_item import DownloadedDatasetItem

    # Enum types
    from sagent.models.environment_type import EnvironmentType

    # Job models
    from sagent.models.job.config import (
        DatasetConfig,
        JobConfig,
        RetryConfig,
    )
    from sagent.models.job.result import AgentDatasetStats, JobResult, JobStats

    # Metric models
    from sagent.models.metric.config import MetricConfig
    from sagent.models.metric.type import MetricType
    from sagent.models.metric.usage_info import UsageInfo

    # Registry models
    from sagent.models.registry import (
        DatasetSpec,
        LocalRegistryInfo,
        Registry,
        RegistryTaskId,
        RemoteRegistryInfo,
    )

    # Task models
    from sagent.models.task.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        VerifierConfig,
    )
    from sagent.models.task.id import GitTaskId, LocalTaskId
    from sagent.models.task.paths import TaskPaths
    from sagent.models.task.task import Task
    from sagent.models.trial.config import (
        AgentConfig as TrialAgentConfig,
    )
    from sagent.models.trial.config import (
        EnvironmentConfig as TrialEnvironmentConfig,
    )
    from sagent.models.trial.config import (
        TaskConfig as TrialTaskConfig,
    )

    # Trial models
    from sagent.models.trial.config import (
        TrialConfig,
    )
    from sagent.models.trial.config import (
        VerifierConfig as TrialVerifierConfig,
    )
    from sagent.models.trial.paths import TrialPaths
    from sagent.models.trial.result import (
        AgentInfo,
        ExceptionInfo,
        ModelInfo,
        TimingInfo,
        TrialResult,
    )

    # Verifier models
    from sagent.models.verifier.result import VerifierResult
    from sagent.trial.hooks import TrialEvent, TrialHookEvent
    from sagent.trial.queue import TrialQueue
    from sagent.trial.trial import Trial
    from sagent.verifier.verifier import Verifier

__version__ = importlib.metadata.version("sagent")


# Lazy imports to avoid loading heavy dependencies at package import time
_LAZY_IMPORTS = {
    # Core classes
    "Job": ("sagent.job", "Job"),
    "Trial": ("sagent.trial.trial", "Trial"),
    "Task": ("sagent.models.task.task", "Task"),
    "BaseAgent": ("sagent.agents.base", "BaseAgent"),
    "BaseEnvironment": ("sagent.environments.base", "BaseEnvironment"),
    "ExecResult": ("sagent.environments.base", "ExecResult"),
    "Verifier": ("sagent.verifier.verifier", "Verifier"),
    "TrialQueue": ("sagent.trial.queue", "TrialQueue"),
    # Job models
    "JobConfig": ("sagent.models.job.config", "JobConfig"),
    "RetryConfig": ("sagent.models.job.config", "RetryConfig"),
    "DatasetConfig": ("sagent.models.job.config", "DatasetConfig"),
    "LocalDatasetConfig": (
        "sagent.models.job.config",
        "DatasetConfig",
    ),  # deprecated alias
    "RegistryDatasetConfig": (
        "sagent.models.job.config",
        "DatasetConfig",
    ),  # deprecated alias
    "JobResult": ("sagent.models.job.result", "JobResult"),
    "JobStats": ("sagent.models.job.result", "JobStats"),
    "AgentDatasetStats": ("sagent.models.job.result", "AgentDatasetStats"),
    # Trial models
    "TrialConfig": ("sagent.models.trial.config", "TrialConfig"),
    "TrialAgentConfig": ("sagent.models.trial.config", "AgentConfig"),
    "TrialEnvironmentConfig": ("sagent.models.trial.config", "EnvironmentConfig"),
    "TrialVerifierConfig": ("sagent.models.trial.config", "VerifierConfig"),
    "TrialTaskConfig": ("sagent.models.trial.config", "TaskConfig"),
    "TrialResult": ("sagent.models.trial.result", "TrialResult"),
    "TimingInfo": ("sagent.models.trial.result", "TimingInfo"),
    "ExceptionInfo": ("sagent.models.trial.result", "ExceptionInfo"),
    "ModelInfo": ("sagent.models.trial.result", "ModelInfo"),
    "AgentInfo": ("sagent.models.trial.result", "AgentInfo"),
    "TrialPaths": ("sagent.models.trial.paths", "TrialPaths"),
    # Task models
    "TaskConfig": ("sagent.models.task.config", "TaskConfig"),
    "EnvironmentConfig": ("sagent.models.task.config", "EnvironmentConfig"),
    "AgentConfig": ("sagent.models.task.config", "AgentConfig"),
    "VerifierConfig": ("sagent.models.task.config", "VerifierConfig"),
    "TaskPaths": ("sagent.models.task.paths", "TaskPaths"),
    "LocalTaskId": ("sagent.models.task.id", "LocalTaskId"),
    "GitTaskId": ("sagent.models.task.id", "GitTaskId"),
    # Agent models
    "AgentContext": ("sagent.models.agent.context", "AgentContext"),
    "AgentName": ("sagent.models.agent.name", "AgentName"),
    # Verifier models
    "VerifierResult": ("sagent.models.verifier.result", "VerifierResult"),
    # Enum types
    "EnvironmentType": ("sagent.models.environment_type", "EnvironmentType"),
    "TrialEvent": ("sagent.trial.hooks", "TrialEvent"),
    "TrialHookEvent": ("sagent.trial.hooks", "TrialHookEvent"),
    # Registry models
    "LocalRegistryInfo": ("sagent.models.registry", "LocalRegistryInfo"),
    "RemoteRegistryInfo": ("sagent.models.registry", "RemoteRegistryInfo"),
    "RegistryTaskId": ("sagent.models.registry", "RegistryTaskId"),
    "DatasetSpec": ("sagent.models.registry", "DatasetSpec"),
    "Registry": ("sagent.models.registry", "Registry"),
    "DownloadedDatasetItem": ("sagent.models.dataset_item", "DownloadedDatasetItem"),
    # Metric models
    "MetricConfig": ("sagent.models.metric.config", "MetricConfig"),
    "MetricType": ("sagent.models.metric.type", "MetricType"),
    "UsageInfo": ("sagent.models.metric.usage_info", "UsageInfo"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core classes
    "Job",
    "Trial",
    "Task",
    "BaseAgent",
    "BaseEnvironment",
    "ExecResult",
    "Verifier",
    "TrialQueue",
    # Job models
    "JobConfig",
    "RetryConfig",
    "DatasetConfig",
    "LocalDatasetConfig",  # deprecated alias
    "RegistryDatasetConfig",  # deprecated alias
    "JobResult",
    "JobStats",
    "AgentDatasetStats",
    # Trial models
    "TrialConfig",
    "TrialAgentConfig",
    "TrialEnvironmentConfig",
    "TrialVerifierConfig",
    "TrialTaskConfig",
    "TrialResult",
    "TimingInfo",
    "ExceptionInfo",
    "ModelInfo",
    "AgentInfo",
    "TrialPaths",
    # Task models
    "TaskConfig",
    "EnvironmentConfig",
    "AgentConfig",
    "VerifierConfig",
    "TaskPaths",
    "LocalTaskId",
    "GitTaskId",
    # Agent models
    "AgentContext",
    "AgentName",
    # Verifier models
    "VerifierResult",
    # Enum types
    "EnvironmentType",
    "TrialEvent",
    "TrialHookEvent",
    # Registry models
    "LocalRegistryInfo",
    "RemoteRegistryInfo",
    "RegistryTaskId",
    "DatasetSpec",
    "Registry",
    "DownloadedDatasetItem",
    # Metric models
    "MetricConfig",
    "MetricType",
    "UsageInfo",
]
