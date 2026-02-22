"""Tests for rabbitllm.persist (SafetensorModelPersister)."""

import pytest
import torch

from rabbitllm.persist.safetensor import SafetensorModelPersister


def test_model_persist_exist_false_when_no_files(tmp_path):
    persister = SafetensorModelPersister()
    assert not persister.model_persist_exist("model.layers.0", tmp_path)


def test_persist_and_load_model_roundtrip(tmp_path):
    persister = SafetensorModelPersister()
    state_dict = {"weight": torch.randn(2, 2)}
    layer_name = "model.layers.0"
    persister.persist_model(state_dict, layer_name, tmp_path)
    assert persister.model_persist_exist(layer_name, tmp_path)
    loaded = persister.load_model(layer_name, tmp_path)
    assert "weight" in loaded
    torch.testing.assert_close(loaded["weight"], state_dict["weight"])


def test_load_model_raises_for_missing_layer(tmp_path):
    persister = SafetensorModelPersister()
    with pytest.raises(Exception):
        persister.load_model("nonexistent.layer", tmp_path)
