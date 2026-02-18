"""Attention implementation resolution and meta model creation."""

import contextlib
import io
import logging
import warnings

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

ATTN_FALLBACK_ORDER = {
    "flash_attention_2": ["flash_attention_2", "sdpa", "eager"],
    "sdpa": ["sdpa", "eager"],
    "eager": ["eager"],
}


def resolve_attn_implementation(dtype, attn_implementation, is_flash_available_fn):
    """Resolve the best attention implementation to use.

    Parameters
    ----------
    dtype : torch.dtype
        Model running dtype.
    attn_implementation : str
        User request: "auto", "flash_attention_2", "sdpa", or "eager".
    is_flash_available_fn : callable
        Function that returns (ok: bool, message: str).

    Returns
    -------
    str
        One of "flash_attention_2", "sdpa", "eager".
    """
    if attn_implementation != "auto":
        return attn_implementation

    if dtype not in (torch.float16, torch.bfloat16):
        logger.info(
            "dtype %s is not compatible with FlashAttention 2 (requires fp16/bf16). Using SDPA.",
            dtype,
        )
        return "sdpa"

    flash_ok, flash_msg = is_flash_available_fn()
    if flash_ok:
        logger.info(flash_msg)
        return "flash_attention_2"

    logger.info("FlashAttention not available: %s. Using SDPA.", flash_msg)
    return "sdpa"


def _config_with_canonical_head_dim(config):
    """Return a new config with head_dim = hidden_size // num_attention_heads (for Qwen2/5.2 RoPE)."""
    if not hasattr(config, "hidden_size") or not hasattr(config, "num_attention_heads"):
        return config
    canonical = config.hidden_size // config.num_attention_heads
    if getattr(config, "head_dim", None) == canonical:
        return config
    # New config from dict so the model is created with correct head_dim (transformers 5.2).
    d = config.to_dict()
    d["head_dim"] = canonical
    return type(config)(**d)


def create_model_from_config(config, attn_implementation, **extra_kwargs):
    """Create a meta model, suppressing noisy output from third-party model code (e.g. QWen).

    Some model implementations (notably QWen v1) emit flash-attn import warnings
    via print(), logging, and warnings on every instantiation. We suppress all three
    channels during model creation to keep output clean.

    Parameters
    ----------
    config : PretrainedConfig
        HuggingFace model config.
    attn_implementation : str
        Attention implementation to use.
    **extra_kwargs
        Passed to AutoModelForCausalLM.from_config.

    Returns
    -------
    PreTrainedModel
        Model on meta device.
    """
    config = _config_with_canonical_head_dim(config)
    devnull = io.StringIO()
    with (
        init_empty_weights(),
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        root_logger = logging.getLogger()
        prev_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        try:
            return AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                **extra_kwargs,
            )
        finally:
            root_logger.setLevel(prev_level)
