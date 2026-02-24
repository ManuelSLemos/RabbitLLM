"""Disk-backed KV cache for long-context inference.

Offloads KV cache to SSD to support 50k+ token contexts without OOM.
Use when kv_cache_dir is passed to AutoModel.from_pretrained().
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import torch

from transformers import DynamicCache

logger = logging.getLogger(__name__)


class DiskKVCache(DynamicCache):
    """KV cache that offloads K/V to disk after each layer update, keeping GPU usage minimal.

    A single instance is shared across all decoder layers during one generation run.
    On each ``update()`` call:

    1. Loads the previous K/V for that layer from disk (if a prior step saved it).
    2. Concatenates with the incoming K/V tensors (adding the current token).
    3. Saves the combined K/V to disk (``layer_{layer_idx}.pt``).
    4. Returns the combined K/V for the attention computation.
    5. Does **not** store tensors in ``self.layers`` — GPU is freed after the call.

    This mirrors the oLLM approach: GPU holds at most one layer's K/V at a time during
    the forward pass, and all context accumulates on disk rather than in VRAM.

    Usage in RabbitLLM: pass ``kv_cache_dir`` to ``AutoModel.from_pretrained()``; one
    ``DiskKVCache`` object is created at the start of generation and reused as
    ``past_key_values`` across all decode steps.
    """

    def __init__(
        self,
        cache_dir: str,
        device: str = "cuda:0",
        reset: bool = True,
    ) -> None:
        """
        Args:
            cache_dir: Root directory for the disk cache. A ``kv_cache/`` subdirectory
                is created inside it.
            device: Device to load tensors onto when reading from disk.
            reset: If True (default), delete and recreate the cache directory. Pass
                False to reuse files from a previous run (not recommended; for testing).
        """
        super().__init__()
        if not cache_dir:
            raise ValueError(
                "cache_dir cannot be empty. For in-memory cache, use past_key_values=None"
            )
        self._cache_dir = os.path.join(cache_dir, "kv_cache")
        self._device = device
        # Sequence length tracked manually since self.layers stays empty.
        self._seq_len: int = 0
        if reset:
            if os.path.exists(self._cache_dir):
                shutil.rmtree(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _path(self, layer_idx: int) -> str:
        return os.path.join(self._cache_dir, f"layer_{layer_idx}.pt")

    # ------------------------------------------------------------------
    # DynamicCache API overrides
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        # Make ``if past_key_values:`` evaluate to True once seq_len is set, even
        # though self.layers is empty (and DynamicCache.__len__ would return 0).
        return self._seq_len > 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return total cached sequence length.

        Tracked manually because ``self.layers`` stays empty (GPU is cleared after
        every ``update()`` call to avoid VRAM accumulation).
        """
        return self._seq_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load previous K/V from disk, append new states, save back, return combined.

        This method replaces the DynamicCache in-GPU storage with disk I/O so that
        VRAM is not accumulated across layers or decode steps.

        Args:
            key_states: New key tensor for this step (batch, heads, seq, head_dim).
            value_states: New value tensor for this step.
            layer_idx: Real decoder layer index (0-based). The caller must set the
                attention module's ``layer_idx`` to this value before calling the
                layer forward (via ``_layer_idx_set``).
            cache_kwargs: Unused; kept for API compatibility with DynamicCache.

        Returns:
            (k_full, v_full): Combined K/V including all prior context and the current
                step, on the same device as ``key_states``. Used by the attention
                computation for this layer and released afterwards (not retained).
        """
        p = self._path(layer_idx)
        if os.path.exists(p):
            try:
                k_prev, v_prev = torch.load(p, map_location=self._device, weights_only=True)
                k_full = torch.cat([k_prev, key_states], dim=-2)
                v_full = torch.cat([v_prev, value_states], dim=-2)
            except Exception as e:
                logger.warning(
                    "KV cache load failed for layer %s: %s — treating as fresh", layer_idx, e
                )
                k_full = key_states
                v_full = value_states
        else:
            k_full = key_states
            v_full = value_states

        try:
            torch.save((k_full.cpu(), v_full.cpu()), p)
        except Exception as e:
            logger.warning("KV cache save failed for layer %s: %s", layer_idx, e)

        # Update sequence length (all layers share the same seq len, last write wins).
        self._seq_len = k_full.shape[-2]

        # Return for attention computation.  Do NOT store in self.layers — that
        # would keep the large tensors in VRAM, defeating the purpose of disk offload.
        return k_full, v_full
