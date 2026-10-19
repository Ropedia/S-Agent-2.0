from pathlib import Path

from sagent.constants import DEFAULT_REGISTRY_URL
from sagent.registry.client.base import BaseRegistryClient
from sagent.registry.client.sagent.sagent import SAgentRegistryClient
from sagent.registry.client.json import JsonRegistryClient


class RegistryClientFactory:
    @staticmethod
    def create(
        registry_url: str | None = None,
        registry_path: Path | None = None,
    ) -> BaseRegistryClient:
        if registry_path is not None:
            return JsonRegistryClient(path=registry_path)
        elif registry_url is not None:
            if registry_url == DEFAULT_REGISTRY_URL:
                return SAgentRegistryClient()
            return JsonRegistryClient(url=registry_url)
        else:
            return SAgentRegistryClient()
