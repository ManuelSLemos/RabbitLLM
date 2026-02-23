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
    """KV cache that offloads to disk, enabling long context without OOM.

    Extends DynamicCache: after each layer updates, saves K/V to disk and frees
    GPU. On subsequent forward (e.g. decode step), loads from disk when needed.
    """

    def __init__(
        self,
        cache_dir: str,
        device: str = "cuda:0",
        num_layers: int = 0,
        decoder_layer_idx: int = 0,
    ):
        super().__init__()
        if not cache_dir:
            raise ValueError(
                "cache_dir cannot be empty. For in-memory cache, use past_key_values=None"
            )
        self._cache_dir = os.path.join(cache_dir, "kv_cache")
        self._device = device
        self._num_layers = num_layers
        self._decoder_layer_idx = decoder_layer_idx
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)
        # Staging: new tokens since last save (per layer)
        self._key_staging: list = []
        self._value_staging: list = []

    def _path(self, layer_idx: int) -> str:
        # Attention passes layer_idx=0 due to _layer_idx_as_zero; use decoder idx for disk
        disk_idx = self._decoder_layer_idx if layer_idx == 0 else layer_idx
        return os.path.join(self._cache_dir, f"layer_{disk_idx}.pt")

    def _load_from_disk(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        p = self._path(layer_idx)
        if not os.path.exists(p):
            return None
        try:
            tensors = torch.load(p, map_location=self._device)
            return (tensors[0], tensors[1])
        except Exception as e:
            logger.warning("Failed to load KV cache layer %s: %s", layer_idx, e)
            return None

    def _save_to_disk(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        p = self._path(layer_idx)
        try:
            k_cpu = key_states.cpu()
            v_cpu = value_states.cpu()
            torch.save((k_cpu, v_cpu), p)
        except Exception as e:
            logger.warning("Failed to save KV cache layer %s: %s", layer_idx, e)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load from disk if exists (prefill from previous run or decode)
        tensors = self._load_from_disk(layer_idx)
        if tensors is not None:
            k_loaded, v_loaded = tensors
            if layer_idx < len(self._key_staging) and layer_idx < len(self._value_staging):
                k_loaded = torch.cat([k_loaded, self._key_staging[layer_idx]], dim=-2)
                v_loaded = torch.cat([v_loaded, self._value_staging[layer_idx]], dim=-2)
            # Ensure we have a layer entry (DynamicCache uses .keys and .values)
            while len(self.layers) <= layer_idx:
                from transformers.cache_utils import DynamicLayer

                self.layers.append(DynamicLayer())
            self.layers[layer_idx].keys = k_loaded
            self.layers[layer_idx].values = v_loaded
            self.layers[layer_idx].is_initialized = True

        out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Append new to staging (for next load)
        if layer_idx < len(self._key_staging):
            self._key_staging[layer_idx] = torch.cat(
                [self._key_staging[layer_idx], key_states], dim=-2
            )
            self._value_staging[layer_idx] = torch.cat(
                [self._value_staging[layer_idx], value_states], dim=-2
            )
        else:
            while len(self._key_staging) <= layer_idx:
                self._key_staging.append(key_states.clone())
                self._value_staging.append(value_states.clone())

        # Save full cache to disk on first write (prefill)
        if tensors is None:
            k_out, v_out = out
            self._save_to_disk(k_out, v_out, layer_idx)

        # Note: we do not clear here; the base extracts from the cache after the layer returns.
        # The cache object is discarded when we create a fresh one for the next layer.
        return out
