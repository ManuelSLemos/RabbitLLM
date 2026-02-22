"""Tests for rabbitllm.utils.splitting."""

import shutil
from unittest.mock import MagicMock

import pytest
import torch

from rabbitllm.utils.memory import NotEnoughSpaceException
from rabbitllm.utils.splitting import (
    check_space,
    load_layer,
    remove_real_and_linked_file,
)


def test_load_layer_returns_state_dict_from_persister():
    persister = MagicMock()
    persister.load_model.return_value = {"weight": torch.randn(2, 2)}
    result = load_layer("/fake/path", "model.layers.0", persister=persister)
    assert isinstance(result, dict)
    assert "weight" in result
    persister.load_model.assert_called_once_with("model.layers.0", "/fake/path")


def test_load_layer_with_profiling_returns_tuple():
    persister = MagicMock()
    persister.load_model.return_value = {"weight": torch.randn(2, 2)}
    result = load_layer("/fake/path", "model.layers.0", profiling=True, persister=persister)
    assert isinstance(result, tuple)
    state_dict, elapsed = result
    assert isinstance(state_dict, dict)
    assert isinstance(elapsed, (int, float))


def test_check_space_passes_when_enough_space(tmp_path):
    (tmp_path / "dummy.bin").write_bytes(b"x" * 100)
    check_space(tmp_path)


def test_check_space_raises_when_not_enough_space(tmp_path):
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1024)
    original_disk_usage = shutil.disk_usage

    def mock_disk_usage(_p):
        return (1000, 1000, 0)

    shutil.disk_usage = mock_disk_usage
    try:
        with pytest.raises(NotEnoughSpaceException):
            check_space(tmp_path)
    finally:
        shutil.disk_usage = original_disk_usage


def test_remove_real_and_linked_file_removes_file(tmp_path):
    path = tmp_path / "to_delete.bin"
    path.write_bytes(b"x")
    assert path.exists()
    remove_real_and_linked_file(path)
    assert not path.exists()
