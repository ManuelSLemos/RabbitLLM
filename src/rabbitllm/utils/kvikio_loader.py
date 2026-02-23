from __future__ import annotations

"""GPU Direct Storage (kvikio) loader for safetensors.

When kvikio is available, reads layer safetensors directly from disk to GPU,
bypassing the CPU and eliminating pin_memory. Optional: pip install rabbitllm[gds]
"""

import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

kvikio_available = False
try:
    import kvikio  # noqa: F401
    import cupy as cp

    kvikio_available = True
except ImportError:
    cp = None  # type: ignore

# Safetensors dtype codes to torch dtypes
_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "I64": torch.int64,
}


def _is_kvikio_usable(device: Optional[str] = None) -> bool:
    """Return True if kvikio can be used (installed + CUDA device)."""
    if not kvikio_available:
        return False
    if device is None or not device.startswith("cuda"):
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def load_safetensor_layer_to_gpu(
    local_path: str,
    layer_name: str,
    device: str,
    dtype: Optional[torch.dtype] = None,
    persister: Optional[Any] = None,
) -> Dict[str, torch.Tensor]:
    """Load a layer safetensor directly to GPU via kvikio (GPU Direct Storage).

    Bypasses CPU, pin_memory, and CPU→GPU transfer. Requires kvikio and cupy.
    Falls back to standard load if kvikio unavailable or layer uses compression.

    Args:
        local_path: Path to split checkpoint directory.
        layer_name: Layer key (e.g. "model.layers.0").
        device: Target device (e.g. "cuda:0").
        dtype: Optional target dtype; if None, uses dtype from safetensor.
        persister: Ignored (used for API compatibility).

    Returns:
        state_dict with tensors on device, or empty dict on fallback.
    """
    if not _is_kvikio_usable(device):
        return {}

    base = Path(local_path)
    for suffix in (layer_name + "safetensors", layer_name + ".safetensors"):
        filepath = base / suffix
        if not filepath.exists():
            continue

        try:
            return _load_safetensor_file_to_gpu(str(filepath), device, dtype)
        except Exception as e:
            logger.warning("kvikio load failed for %s: %s. Falling back to CPU path.", filepath, e)
            return {}
    return {}


def _load_safetensor_file_to_gpu(
    filepath: str,
    device: str,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load a single safetensor file to GPU via kvikio.CuFile."""
    if not kvikio_available:
        return {}

    # Read header with regular Python (header is small, metadata)
    with open(filepath, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_offset = 8 + header_len

    # Per-tensor metadata: {"__metadata__": {...}} or {"weight": {"dtype": "F16", ...}}
    result: Dict[str, torch.Tensor] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(meta, dict) or "dtype" not in meta or "shape" not in meta:
            continue
        if "data_offsets" not in meta:
            continue

        off0, off1 = meta["data_offsets"]
        nbytes = off1 - off0
        shape = tuple(meta["shape"])
        dtype_str = meta["dtype"]

        if dtype_str not in _DTYPE_MAP:
            logger.debug("Skipping tensor %s: unsupported dtype %s", key, dtype_str)
            continue

        torch_dtype = dtype or _DTYPE_MAP[dtype_str]

        # Load raw bytes to GPU (works for all dtypes including BF16)
        dev_idx = 0
        if device.startswith("cuda:") and len(device) > 5:
            try:
                dev_idx = int(device.split(":")[1])
            except ValueError:
                pass
        with cp.cuda.Device(dev_idx):
            buf = cp.empty((nbytes,), dtype=cp.uint8)

        with kvikio.CuFile(filepath, "r") as kvf:
            future = kvf.pread(buf, file_offset=data_offset + off0)
            nread = future.get()
        if nread != nbytes:
            raise IOError(f"Short read: {nread} of {nbytes} bytes for {key}")

        # CuPy uint8 buffer to PyTorch, view as target dtype
        t = torch.as_tensor(buf, device=device).view(torch.uint8)
        t = t.view(_DTYPE_MAP[dtype_str]).reshape(shape).contiguous()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        result[key] = t

    return result
