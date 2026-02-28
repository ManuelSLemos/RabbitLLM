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
            DiskKVCache(cache_dir="", device="cpu")
        assert "empty" in str(ctx.exception).lower()

    def test_update_saves_to_disk(self):
        """update() persists K/V to disk and returns correct shape."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(cache_dir=tmp, device="cpu")
            k = torch.randn(1, 2, 4, 8)
            v = torch.randn(1, 2, 4, 8)
            out_k, out_v = cache.update(k, v, 0)
            assert out_k.shape == k.shape, "First update: shape should match input"
            assert out_v.shape == v.shape
            assert os.path.exists(os.path.join(tmp, "kv_cache", "layer_0.pt"))

    def test_update_tracks_seq_len(self):
        """get_seq_length() reflects number of cached tokens after update()."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(cache_dir=tmp, device="cpu")
            k = torch.randn(1, 2, 5, 8)
            v = torch.randn(1, 2, 5, 8)
            cache.update(k, v, layer_idx=0)
            assert cache.get_seq_length() == 5
            assert bool(cache)  # __bool__ returns True once seq_len > 0

    def test_persisted_cache_has_correct_shape(self):
        """After update(), persisted file contains tensors with expected shape."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(cache_dir=tmp, device="cpu")
            k1 = torch.randn(1, 2, 2, 8)
            v1 = torch.randn(1, 2, 2, 8)
            cache.update(k1, v1, 0)
            layer_path = os.path.join(tmp, "kv_cache", "layer_0.pt")
            assert os.path.exists(layer_path)
            loaded = torch.load(layer_path, map_location="cpu", weights_only=True)
            assert loaded[0].shape == (1, 2, 2, 8)

    def test_multiple_layers(self):
        """Different layer indices save to separate files."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(cache_dir=tmp, device="cpu")
            for i in range(2):
                k = torch.randn(1, 2, 1, 8)
                v = torch.randn(1, 2, 1, 8)
                cache.update(k, v, i)
            kv_dir = os.path.join(tmp, "kv_cache")
            assert os.path.exists(os.path.join(kv_dir, "layer_0.pt"))
            assert os.path.exists(os.path.join(kv_dir, "layer_1.pt"))

    def test_decode_appends_token(self):
        """Second update() for the same layer appends the new token to the saved K/V."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = DiskKVCache(cache_dir=tmp, device="cpu")
            # Prefill: 3 tokens
            k1 = torch.randn(1, 2, 3, 8)
            v1 = torch.randn(1, 2, 3, 8)
            cache.update(k1, v1, layer_idx=0)
            assert cache.get_seq_length() == 3

            # Decode: 1 new token
            k2 = torch.randn(1, 2, 1, 8)
            v2 = torch.randn(1, 2, 1, 8)
            out_k, out_v = cache.update(k2, v2, layer_idx=0)
            assert out_k.shape[-2] == 4, "Expected prefill(3) + new(1) = 4 tokens"
            assert out_v.shape[-2] == 4
            assert cache.get_seq_length() == 4

    def test_reset_clears_old_files(self):
        """DiskKVCache(reset=True) deletes files from a previous cache."""
        with tempfile.TemporaryDirectory() as tmp:
            cache1 = DiskKVCache(cache_dir=tmp, device="cpu", reset=True)
            cache1.update(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4), layer_idx=0)
            assert os.path.exists(os.path.join(tmp, "kv_cache", "layer_0.pt"))

            # New cache with reset=True should wipe old files
            cache2 = DiskKVCache(cache_dir=tmp, device="cpu", reset=True)
            assert not os.path.exists(
                os.path.join(tmp, "kv_cache", "layer_0.pt")
            ), "reset=True should delete old cache files"
            assert cache2.get_seq_length() == 0
            assert not bool(cache2)  # __bool__ is False when seq_len == 0


@pytest.mark.skipif(DynamicCache is None, reason="transformers.cache_utils.DynamicCache required")
class TestDiskKVCacheIntegration(unittest.TestCase):
    """Integration: base model uses DiskKVCache when kv_cache_dir is set."""

    def test_make_layer_past_kv_arg_returns_disk_cache_when_active(self):
        """When _active_disk_kv_cache is set, _make_layer_past_kv_arg passes it through."""
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_bare_model()
            model.kv_cache_dir = tmp
            model.running_device = "cpu"
            # In the new design the shared cache is created in forward() and stored here.
            model._active_disk_kv_cache = DiskKVCache(cache_dir=tmp, device="cpu")

            out = model._make_layer_past_kv_arg(decoder_layer_idx=0)
            assert "past_key_value" in out
            assert out["past_key_value"] is model._active_disk_kv_cache

    def test_make_layer_past_kv_arg_returns_dynamic_cache_when_inactive(self):
        """When _active_disk_kv_cache is None, _make_layer_past_kv_arg returns a DynamicCache."""
        model = _make_bare_model()
        model._active_disk_kv_cache = None
        out = model._make_layer_past_kv_arg(decoder_layer_idx=0)
        assert "past_key_value" in out
        assert isinstance(out["past_key_value"], DynamicCache)
        assert not isinstance(out["past_key_value"], DiskKVCache)
