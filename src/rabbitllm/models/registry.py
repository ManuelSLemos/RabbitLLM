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

        arch = config.architectures[0]
        if "Qwen2ForCausalLM" in arch or "Qwen2.5" in arch:
            return "rabbitllm.models.qwen2", "RabbitLLMQWen2"
        if "Qwen3" in arch:
            return "rabbitllm.models.qwen3", "RabbitLLMQWen3"
        if "QWen" in arch:
            return "rabbitllm.models.qwen", "RabbitLLMQWen"
        if "Baichuan" in arch:
            return "rabbitllm.models.baichuan", "RabbitLLMBaichuan"
        if "ChatGLM" in arch:
            return "rabbitllm.models.chatglm", "RabbitLLMChatGLM"
        if "InternLM" in arch:
            return "rabbitllm.models.internlm", "RabbitLLMInternLM"
        if "Mistral" in arch and "Mixtral" not in arch:
            return "rabbitllm.models.mistral", "RabbitLLMMistral"
        if "Mixtral" in arch:
            return "rabbitllm.models.mixtral", "RabbitLLMMixtral"
        if "Gemma" in arch or "Phi2" in arch or "Phi3" in arch or "Phi4" in arch:
            return "rabbitllm.models.llama", "RabbitLLMLlama2"
        if "DeepSeek" in arch:
            return "rabbitllm.models.llama", "RabbitLLMLlama2"
        if "Llama" in arch:
            return "rabbitllm.models.llama", "RabbitLLMLlama2"
        else:
            logger.warning(
                "unknown architecture: %s, try to use Llama2...", arch
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
