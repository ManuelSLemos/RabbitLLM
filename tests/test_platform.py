"""Tests for rabbitllm.utils.platform."""

from rabbitllm.utils.platform import (
    is_cuda_available,
    is_flash_attention_available,
    is_macos,
)


def test_is_macos_returns_bool():
    assert isinstance(is_macos(), bool)


def test_is_cuda_available_returns_bool():
    assert isinstance(is_cuda_available(), bool)


def test_is_flash_attention_available_returns_tuple():
    result = is_flash_attention_available()
    assert isinstance(result, tuple)
    assert len(result) == 2
    available, message = result
    assert isinstance(available, bool)
    assert isinstance(message, str)
