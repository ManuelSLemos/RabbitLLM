"""Thin model subclasses for architectures that need only minor overrides.

``RabbitLLMLlama2`` is a plain alias — Llama-style models work with the base
engine without any changes.

``RabbitLLMMistral`` and ``RabbitLLMInternLM`` only differ from the base in
their ``get_generation_config()`` implementation: both return a bare
``GenerationConfig()`` instead of reading one from the checkpoint.  This
avoids errors on checkpoints that don't ship a ``generation_config.json``.
"""

from transformers import GenerationConfig

from ..engine.base import RabbitLLMBaseModel


class RabbitLLMLlama2(RabbitLLMBaseModel):
    """Llama / Gemma / Phi / DeepSeek — uses base engine without modification."""


class RabbitLLMMistral(RabbitLLMBaseModel):
    """Mistral — identical to base except generation config is always bare."""

    def get_generation_config(self):
        return GenerationConfig()


class RabbitLLMInternLM(RabbitLLMBaseModel):
    """InternLM — identical to base except generation config is always bare."""

    def get_generation_config(self):
        return GenerationConfig()
