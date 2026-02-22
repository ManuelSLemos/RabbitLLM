from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

_default_persister: Optional["ModelPersister"] = None


class ModelPersister(ABC):
    """Abstract base for persisting and loading model layers (safetensors or MLX).

    Use get_model_persister() to obtain the platform-appropriate default instance,
    or pass a custom persister to RabbitLLMBaseModel.__init__(persister=...).
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def get_model_persister(cls) -> "ModelPersister":
        """Return the default ModelPersister for this platform.

        Uses Safetensor on Linux/Windows, MLX on macOS.
        """
        global _default_persister
        if _default_persister is not None:
            return _default_persister
        from ..utils.platform import is_on_mac_os

        if is_on_mac_os:
            from .mlx import MlxModelPersister

            _default_persister = MlxModelPersister()  # noqa: PLW0603
        else:
            from .safetensor import SafetensorModelPersister

            _default_persister = SafetensorModelPersister()  # noqa: PLW0603
        return _default_persister

    @abstractmethod
    def model_persist_exist(self, layer_name: str, saving_path: Path) -> bool:
        """Return True if the layer is already persisted at saving_path."""
        ...

    @abstractmethod
    def persist_model(self, state_dict: Dict[str, Any], layer_name: str, path: Path) -> None:
        """Save a layer state_dict under path with the layer name."""
        ...

    @abstractmethod
    def load_model(self, layer_name: str, path: Union[Path, str]) -> Any:
        """Load a layer state_dict from path. Returns dict or MLX structure."""
        ...
