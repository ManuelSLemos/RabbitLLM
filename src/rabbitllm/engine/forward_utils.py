"""Helpers for the layer-streaming forward pass (attention mask, position ids, KV extraction)."""

from typing import Any, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def build_attention_mask_and_position_ids(
    device: str,
    dtype: torch.dtype,
    max_seq_len: int,
    attn_implementation: str,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Build the attention mask and position_ids used at the start of forward().

    - FlashAttention 2: 2D mask (batch, seq_len), causality handled internally.
    - SDPA: mask is None (is_causal=True used by backend).
    - Eager: 4D causal additive mask.

    Parameters
    ----------
    device : str
        Target device (e.g. "cuda:0").
    dtype : torch.dtype
        Model running dtype.
    max_seq_len : int
        Max sequence length.
    attn_implementation : str
        One of "flash_attention_2", "sdpa", "eager".

    Returns
    -------
    attention_mask : tensor or None
        Mask to pass to layers, or None for SDPA.
    position_ids : tensor
        (1, max_seq_len) on device.
    """
    if attn_implementation == "flash_attention_2":
        attention_mask = torch.ones(1, max_seq_len, dtype=torch.long, device=device)
    elif attn_implementation == "sdpa":
        attention_mask = None
    else:
        attention_mask = torch.full(
            (max_seq_len, max_seq_len),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)[None, None, ...]

    position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device)[None, :]
    return attention_mask, position_ids


def _get_kv_from_dynamic_cache(
    cache: Any,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract (k, v) from a DynamicCache using either the 5.x (.layers) or legacy (.key_cache) API.

    In transformers 5.x, DynamicCache stores KV in `.layers[i].keys / .values`.
    In older versions (< 5.x), it used `.key_cache[-1] / .value_cache[-1]`.
    """
    # New API: transformers 5.x
    layers = getattr(cache, "layers", None)
    if layers and len(layers) > 0:
        layer0 = layers[0]
        k = getattr(layer0, "keys", None)
        v = getattr(layer0, "values", None)
        if k is not None and v is not None and k.numel() > 0:
            return k, v
    # Legacy API: transformers < 5.x
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache and value_cache and len(key_cache) > 0:
        return key_cache[-1], value_cache[-1]
    return None, None


def extract_kv_from_layer_output(
    layer_out: Any,
    output_attentions: bool = False,
    cache_utils_installed: bool = False,
    cache_class: Optional[type] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract (hidden_states, k_cache, v_cache) from a decoder layer output.

    Handles both legacy tuple format (eager) and Cache objects (SDPA/FA2).
    In 4.47+, some layers (e.g. Qwen2) return only a tensor; caller may get
    KV from the DynamicCache passed in kwargs.

    Parameters
    ----------
    layer_out : tensor or tuple
        Single tensor (hidden_states only) or tuple (hidden_states, ..., cache_data).
    output_attentions : bool
        Whether attentions are in the tuple (affects cache index).
    cache_utils_installed : bool
        Whether transformers.cache_utils is available.
    cache_class : type, optional
        Cache base class for isinstance (e.g. from transformers.cache_utils).

    Returns
    -------
    hidden_states : tensor
    k_cache : tensor or None
    v_cache : tensor or None
    """
    if isinstance(layer_out, torch.Tensor):
        return layer_out, None, None
    hidden_states = layer_out[0]
    cache_idx = 2 if output_attentions else 1
    cache_data = layer_out[cache_idx] if len(layer_out) > cache_idx else None

    if cache_data is None:
        return hidden_states, None, None

    if isinstance(cache_data, tuple):
        return hidden_states, cache_data[0], cache_data[1]

    if cache_utils_installed and cache_class is not None and isinstance(cache_data, cache_class):
        k, v = _get_kv_from_dynamic_cache(cache_data)
        return hidden_states, k, v

    return hidden_states, None, None
