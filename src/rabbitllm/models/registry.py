from __future__ import annotations

import importlib
import logging
from typing import Any, Tuple

from transformers import AutoConfig

from ..utils.platform import is_on_mac_os

logger = logging.getLogger(__name__)

if is_on_mac_os:
    from ..engine.mlx_engine import RabbitLLMLlamaMlx


class AutoModel:
    """Factory to load the correct RabbitLLM model class from a checkpoint or repo ID.

    Use from_pretrained(); do not instantiate directly.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(
        cls, pretrained_model_name_or_path: str, *inputs: Any, **kwargs: Any
    ) -> Tuple[str, str]:
        """Resolve (module_name, class_name) from model config architectures.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path.
            *inputs: Passed through to AutoConfig.from_pretrained.
            **kwargs: Passed through; hf_token used for gated repos.

        Returns:
            Tuple of (module_name, class_name) e.g. ("rabbitllm.models.llama", "RabbitLLMLlama2").
        """
        if "hf_token" in kwargs:
            logger.debug("using hf_token")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True, token=kwargs["hf_token"]
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )

        if "Qwen2ForCausalLM" in config.architectures[0]:
            return "rabbitllm.models.qwen2", "RabbitLLMQWen2"
        elif "QWen" in config.architectures[0]:
            return "rabbitllm.models.qwen", "RabbitLLMQWen"
        elif "Baichuan" in config.architectures[0]:
            return "rabbitllm.models.baichuan", "RabbitLLMBaichuan"
        elif "ChatGLM" in config.architectures[0]:
            return "rabbitllm.models.chatglm", "RabbitLLMChatGLM"
        elif "InternLM" in config.architectures[0]:
            return "rabbitllm.models.internlm", "RabbitLLMInternLM"
        elif "Mistral" in config.architectures[0]:
            return "rabbitllm.models.mistral", "RabbitLLMMistral"
        elif "Mixtral" in config.architectures[0]:
            return "rabbitllm.models.mixtral", "RabbitLLMMixtral"
        elif "Llama" in config.architectures[0]:
            return "rabbitllm.models.llama", "RabbitLLMLlama2"
        else:
            logger.warning(
                "unknown architecture: %s, try to use Llama2...", config.architectures[0]
            )
            return "rabbitllm.models.llama", "RabbitLLMLlama2"

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        """Load a RabbitLLM model from a checkpoint or HuggingFace repo.

        On macOS uses MLX (Llama only); otherwise uses PyTorch layer-streaming.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path.
            *inputs: Passed to the model constructor.
            **kwargs: Passed to the model constructor (device, compression, hf_token, etc.).

        Returns:
            A RabbitLLM model instance (e.g. RabbitLLMLlama2, RabbitLLMQWen2).
        """
        if is_on_mac_os:
            return RabbitLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module_name, class_name = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
