"""Tests for rabbitllm.engine.kvcache (DiskKVCache).

DiskKVCache offloads KV cache to disk for long-context inference (50k+ tokens).
"""

from __future__ import annotations

import os
import tempfile
import unittest
import pytest
import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

from rabbitllm.engine.kvcache import DiskKVCache
from rabbitllm.models.llama import RabbitLLMLlama2


def _make_bare_model():
    """Create RabbitLLMLlama2 without __init__."""
    return object.__new__(RabbitLLMLlama2)


@pytest.mark.skipif(DynamicCache is None, reason="transformers.cache_utils.DynamicCache required")
class TestDiskKVCache(unittest.TestCase):
    """Tests for DiskKVCache."""

    def test_empty_cache_dir_raises(self):
        """DiskKVCache raises ValueError for empty cache_dir."""
        with self.assertRaises(ValueError) as ctx:
            DiskKVCache(cache_dir="", device="cpu", num_layers=1)
        assert "empty" in str(ctx.exception).lower()

    def test_update_saves_to_disk(self):
        """update() persists K/V to disk and can reload."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(
                cache_dir=tmp,
                device="cpu",
                num_layers=2,
                decoder_layer_idx=0,
            )
            k = torch.randn(1, 2, 4, 8)
            v = torch.randn(1, 2, 4, 8)
            out_k, out_v = cache.update(k, v, 0)
            assert out_k.shape == k.shape
            assert out_v.shape == v.shape
            assert os.path.exists(os.path.join(tmp, "kv_cache", "layer_0.pt"))

    def test_persisted_cache_has_correct_shape(self):
        """After update(), persisted file contains tensors with expected shape."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(
                cache_dir=tmp,
                device="cpu",
                num_layers=2,
                decoder_layer_idx=0,
            )
            k1 = torch.randn(1, 2, 2, 8)
            v1 = torch.randn(1, 2, 2, 8)
            cache.update(k1, v1, 0)
            layer_path = os.path.join(tmp, "kv_cache", "layer_0.pt")
            assert os.path.exists(layer_path)
            loaded = torch.load(layer_path, map_location="cpu")
            assert loaded[0].shape == (1, 2, 2, 8)

    def test_multiple_layers(self):
        """Different layer indices save to different files."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(
                cache_dir=tmp,
                device="cpu",
                num_layers=3,
                decoder_layer_idx=0,
            )
            for i in range(2):
                k = torch.randn(1, 2, 1, 8)
                v = torch.randn(1, 2, 1, 8)
                cache.update(k, v, i)
            kv_dir = os.path.join(tmp, "kv_cache")
            assert os.path.exists(os.path.join(kv_dir, "layer_0.pt"))
            assert os.path.exists(os.path.join(kv_dir, "layer_1.pt"))


@pytest.mark.skipif(DynamicCache is None, reason="transformers.cache_utils.DynamicCache required")
class TestDiskKVCacheIntegration(unittest.TestCase):
    """Integration: base model uses DiskKVCache when kv_cache_dir is set."""

    def test_make_layer_past_kv_arg_returns_disk_cache_when_kv_cache_dir_set(self):
        """When kv_cache_dir is set, _make_layer_past_kv_arg uses DiskKVCache."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_bare_model()
            model.kv_cache_dir = tmp
            model.running_device = "cpu"
            k = torch.randn(1, 2, 4, 8)
            v = torch.randn(1, 2, 4, 8)
            out = model._make_layer_past_kv_arg(k, v, decoder_layer_idx=0)
            assert "past_key_value" in out
            assert isinstance(out["past_key_value"], DiskKVCache)
            assert os.path.exists(os.path.join(tmp, "kv_cache", "layer_0.pt"))
