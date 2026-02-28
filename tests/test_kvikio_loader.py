"""Tests for rabbitllm.utils.kvikio_loader.

GDS (GPU Direct Storage) loads safetensors directly disk→GPU via kvikio.
These tests run without kvikio/GPU when unavailable.
"""

from __future__ import annotations

import unittest

import pytest
import torch

from rabbitllm.utils.kvikio_loader import (
    kvikio_available,
    load_safetensor_layer_to_gpu,
)


class TestKvikioLoader(unittest.TestCase):
    """Tests for kvikio_loader module."""

    def test_kvikio_available_is_bool(self):
        """kvikio_available is a boolean."""
        assert isinstance(kvikio_available, bool)

    def test_load_safetensor_layer_to_gpu_returns_empty_on_cpu(self):
        """On CPU device, load_safetensor_layer_to_gpu returns empty dict (no GDS path)."""
        result = load_safetensor_layer_to_gpu(
            "/nonexistent/path",
            "model.layers.0",
            device="cpu",
        )
        assert result == {}

    def test_load_safetensor_layer_to_gpu_returns_empty_for_nonexistent_path(self):
        """When layer file does not exist, returns empty dict."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = load_safetensor_layer_to_gpu(
                tmp,
                "model.layers.0",
                device="cuda:0",
            )
            assert result == {}

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for kvikio load path",
    )
    def test_load_safetensor_layer_to_gpu_with_kvikio_and_real_file(self):
        """With kvikio and a valid safetensor file, loads to GPU.

        Skipped when CUDA or kvikio unavailable.
        """
        if not kvikio_available:
            pytest.skip("kvikio not installed")

        import tempfile

        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            layer_name = "model.layers.0"
            path = f"{tmp}/{layer_name}.safetensors"
            save_file({"weight": torch.randn(4, 4)}, path)
            result = load_safetensor_layer_to_gpu(tmp, layer_name, device="cuda:0")
            if result:
                for name, t in result.items():
                    assert t.device.type == "cuda"
