"""Tests for the RabbitLLM engine: base model, forward utilities, and compression helpers.

All tests run without a GPU or network access. GPU-only tests are marked with
``@pytest.mark.skipif`` so they are skipped in CI.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from rabbitllm import AutoModel
from rabbitllm.engine.forward_utils import (
    _get_kv_from_dynamic_cache,
    build_attention_mask_and_position_ids,
    extract_kv_from_layer_output,
)
from rabbitllm.models.llama import RabbitLLMLlama2
from rabbitllm.utils.compression import compress_layer_state_dict, uncompress_layer_state_dict

# ---------------------------------------------------------------------------
# Forward utilities (no GPU required)
# ---------------------------------------------------------------------------


class TestBuildAttentionMaskAndPositionIds(unittest.TestCase):
    """build_attention_mask_and_position_ids on CPU."""

    def _run(self, attn_impl: str, max_seq_len: int = 8):
        return build_attention_mask_and_position_ids(
            device="cpu",
            dtype=torch.float32,
            max_seq_len=max_seq_len,
            attn_implementation=attn_impl,
        )

    def test_eager_returns_4d_causal_mask(self):
        mask, pos_ids = self._run("eager")
        assert mask is not None
        assert mask.dim() == 4, f"expected 4D mask, got {mask.dim()}D"
        assert mask.shape == (1, 1, 8, 8)
        assert pos_ids.shape == (1, 8)

    def test_sdpa_returns_none_mask(self):
        mask, pos_ids = self._run("sdpa")
        assert mask is None
        assert pos_ids.shape == (1, 8)

    def test_flash_returns_2d_ones_mask(self):
        mask, pos_ids = self._run("flash_attention_2")
        assert mask is not None
        assert mask.dim() == 2
        assert mask.shape == (1, 8)
        assert mask.dtype == torch.long
        assert mask.sum().item() == 8

    def test_position_ids_are_sequential(self):
        _, pos_ids = self._run("sdpa", max_seq_len=16)
        expected = torch.arange(16).unsqueeze(0)
        assert torch.equal(pos_ids, expected)


class TestExtractKvFromLayerOutput(unittest.TestCase):
    """extract_kv_from_layer_output handles all output shapes."""

    def test_tensor_only_output(self):
        hs = torch.randn(1, 4, 8)
        hidden, k, v = extract_kv_from_layer_output(hs)
        assert torch.equal(hidden, hs)
        assert k is None
        assert v is None

    def test_tuple_with_kv_tuple(self):
        hs = torch.randn(1, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = (hs, (k, v))
        hidden, k_out, v_out = extract_kv_from_layer_output(out)
        assert torch.equal(hidden, hs)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_tuple_with_none_cache(self):
        hs = torch.randn(1, 4, 8)
        out = (hs, None)
        hidden, k, v = extract_kv_from_layer_output(out)
        assert torch.equal(hidden, hs)
        assert k is None
        assert v is None

    def test_tuple_without_cache_entry(self):
        hs = torch.randn(1, 4, 8)
        out = (hs,)
        hidden, k, v = extract_kv_from_layer_output(out)
        assert torch.equal(hidden, hs)
        assert k is None
        assert v is None

    def test_output_attentions_shifts_cache_index(self):
        hs = torch.randn(1, 4, 8)
        attn_weights = torch.randn(1, 2, 4, 4)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = (hs, attn_weights, (k, v))
        hidden, k_out, v_out = extract_kv_from_layer_output(out, output_attentions=True)
        assert torch.equal(hidden, hs)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)


class TestGetKvFromDynamicCache(unittest.TestCase):
    """_get_kv_from_dynamic_cache supports both transformers 5.x and legacy APIs."""

    def test_new_api_layers(self):
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        layer0 = MagicMock()
        layer0.keys = k
        layer0.values = v
        cache = MagicMock()
        cache.layers = [layer0]
        k_out, v_out = _get_kv_from_dynamic_cache(cache)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_legacy_key_cache_api(self):
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache = MagicMock()
        cache.layers = []
        cache.key_cache = [k]
        cache.value_cache = [v]
        k_out, v_out = _get_kv_from_dynamic_cache(cache)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

    def test_returns_none_when_empty(self):
        cache = MagicMock()
        cache.layers = []
        cache.key_cache = []
        cache.value_cache = []
        k_out, v_out = _get_kv_from_dynamic_cache(cache)
        assert k_out is None
        assert v_out is None


# ---------------------------------------------------------------------------
# RabbitLLMBaseModel instance methods (no __init__ called)
# ---------------------------------------------------------------------------


def _make_bare_model() -> RabbitLLMLlama2:
    """Create a RabbitLLMLlama2 instance without triggering __init__.

    Used to test methods that depend only on a subset of instance state.
    """
    return object.__new__(RabbitLLMLlama2)


class TestSetLayerNamesDict(unittest.TestCase):
    """set_layer_names_dict populates the expected keys."""

    def test_default_keys_present(self):
        model = _make_bare_model()
        model.set_layer_names_dict()
        required = {"embed", "layer_prefix", "norm", "lm_head"}
        assert required.issubset(model.layer_names_dict.keys())

    def test_default_embed_value(self):
        model = _make_bare_model()
        model.set_layer_names_dict()
        assert model.layer_names_dict["embed"] == "model.embed_tokens"

    def test_default_lm_head_value(self):
        model = _make_bare_model()
        model.set_layer_names_dict()
        assert model.layer_names_dict["lm_head"] == "lm_head"


class TestGetSequenceLen(unittest.TestCase):
    """get_sequence_len handles both 2D and 3D tensors."""

    def test_3d_tensor(self):
        model = _make_bare_model()
        seq = torch.zeros(2, 7, 16)  # (batch, seq_len, hidden)
        assert model.get_sequence_len(seq) == 7

    def test_2d_tensor(self):
        model = _make_bare_model()
        seq = torch.zeros(5, 16)  # (seq_len, hidden)
        assert model.get_sequence_len(seq) == 5


class TestGetPastKeyValuesCacheSeqLen(unittest.TestCase):
    """get_past_key_values_cache_seq_len handles both Cache objects and legacy tuples."""

    def test_legacy_tuple_format(self):
        model = _make_bare_model()
        k = torch.randn(1, 2, 6, 8)
        v = torch.randn(1, 2, 6, 8)
        past = ((k, v),)
        assert model.get_past_key_values_cache_seq_len(past) == 6

    def test_cache_object_format(self):
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            pytest.skip("DynamicCache not available")

        model = _make_bare_model()
        cache = DynamicCache()
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache.update(k, v, 0)
        result = model.get_past_key_values_cache_seq_len(cache)
        assert result == 4


# ---------------------------------------------------------------------------
# AutoModel
# ---------------------------------------------------------------------------

_PATCH_TARGET = "rabbitllm.models.registry.AutoConfig.from_pretrained"


class TestAutoModel(unittest.TestCase):
    """AutoModel public interface."""

    def test_direct_instantiation_raises(self):
        with pytest.raises(EnvironmentError, match="AutoModel is designed to be instantiated"):
            AutoModel()

    def _mock_arch(self, arch: str):
        cfg = MagicMock()
        cfg.architectures = [arch]
        return cfg

    def test_get_module_class_llama(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("LlamaForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/llama-model")
        assert cls == "RabbitLLMLlama2"

    def test_get_module_class_qwen2(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("Qwen2ForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/qwen2-model")
        assert cls == "RabbitLLMQWen2"

    def test_get_module_class_qwen3(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("Qwen3ForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/qwen3-model")
        assert cls == "RabbitLLMQWen3"

    def test_get_module_class_gemma(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("Gemma2ForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/gemma2-model")
        assert cls == "RabbitLLMLlama2"

    def test_get_module_class_deepseek(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("DeepseekV2ForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/deepseek-model")
        assert cls == "RabbitLLMLlama2"

    def test_get_module_class_phi3(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("Phi3ForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/phi3-model")
        assert cls == "RabbitLLMLlama2"

    def test_get_module_class_mistral(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("MistralForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/mistral-model")
        assert cls == "RabbitLLMMistral"

    def test_get_module_class_mixtral(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("MixtralForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/mixtral-model")
        assert cls == "RabbitLLMMixtral"

    def test_get_module_class_chatglm(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("ChatGLMModel")):
            _, cls = AutoModel.get_module_class("fake/chatglm-model")
        assert cls == "RabbitLLMChatGLM"

    def test_get_module_class_unknown_falls_back_to_llama(self):
        with patch(_PATCH_TARGET, return_value=self._mock_arch("SomeUnknownArchForCausalLM")):
            _, cls = AutoModel.get_module_class("fake/unknown-model")
        assert cls == "RabbitLLMLlama2"


# ---------------------------------------------------------------------------
# Compression utilities (no GPU required for pass-through path)
# ---------------------------------------------------------------------------


class TestCompressionPassThrough(unittest.TestCase):
    """compress/uncompress are no-ops when compression=None or dict has no compressed keys."""

    def test_compress_none_returns_same_dict(self):
        state_dict = {"weight": torch.randn(4, 4)}
        result = compress_layer_state_dict(state_dict, compression=None)
        assert result is state_dict

    def test_uncompress_uncompressed_returns_same_dict(self):
        state_dict = {"weight": torch.randn(4, 4)}
        result = uncompress_layer_state_dict(state_dict)
        assert result is state_dict


if __name__ == "__main__":
    unittest.main()
