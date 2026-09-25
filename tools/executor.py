from typing import Protocol

from tools.registry import ToolRegistry


class ToolExecutor(Protocol):
    """Execution boundary used by the bounded orchestrator."""

    def definitions(self) -> list[dict]:
        """Return only tools approved for the model to request."""

    def invoke(self, name: str, raw_arguments: dict) -> dict:
        """Validate and execute one approved tool call."""


class LocalToolExecutor:
    """Execute approved tools directly through the local registry."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def definitions(self) -> list[dict]:
        return self._registry.definitions()

    def invoke(self, name: str, raw_arguments: dict) -> dict:
        return self._registry.invoke(name, raw_arguments)
