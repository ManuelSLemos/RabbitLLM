"""Pre-allocated pinned memory pool for CPU→GPU layer transfers.

Eliminates the per-layer OS page-locking cost (``tensor.pin_memory()`` allocates
new locked pages from the kernel on every call — ~1.7 s/layer at 70B scale).

The pool pays the ``mlock()`` cost **once at startup**, then each layer load only
pays a plain CPU memcpy into a pre-pinned buffer (~50–100 ms/layer), with no OS
involvement.

Usage::

    from rabbitllm.engine.pinned_pool import PinnedMemoryPool, compute_max_layer_bytes

    max_bytes = compute_max_layer_bytes(checkpoint_path, layer_names)
    pool = PinnedMemoryPool(n_slots=3, slot_bytes=max_bytes)

    # In the load path (replaces tensor.pin_memory() per tensor):
    pinned_dict = pool.pin(state_dict)
    # ... move pinned_dict to GPU (async or sync) ...
    # slot is released automatically when pinned_dict is garbage-collected
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import weakref
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class _WeakRefDict(dict):
    """dict subclass that supports weak references (required for weakref.finalize)."""

    __slots__ = ("__weakref__",)


class PinnedMemoryPool:
    """Pre-allocated page-locked memory pool.

    Allocates ``n_slots`` fixed-size pinned buffers at construction time (one-time
    OS cost).  Each call to :meth:`pin` copies a layer's tensors into a free slot
    via ``view.copy_()`` — a fast CPU memcpy with no kernel involvement.

    Slot lifecycle is managed automatically: a ``weakref.finalize`` callback
    returns the slot to the free list when the returned dict is garbage-collected.
    In CPython's reference-counting GC this fires immediately when the last
    reference to the dict is dropped, which happens naturally as the streaming
    pipeline discards each layer's CPU dict after the GPU transfer completes.

    Args:
        n_slots: Number of concurrent slots.  ``2`` covers one in-flight GPU
            copy + one being processed; ``3`` is recommended when dual-prefetch
            is active (two concurrent async copies + one being processed).
        slot_bytes: Size in bytes of each slot.  Must be ≥ the largest layer
            shard.  Use :func:`compute_max_layer_bytes` to compute this.
    """

    def __init__(self, n_slots: int, slot_bytes: int) -> None:
        if slot_bytes <= 0:
            raise ValueError(f"slot_bytes must be > 0, got {slot_bytes}")
        if n_slots <= 0:
            raise ValueError(f"n_slots must be > 0, got {n_slots}")

        self._slot_bytes = slot_bytes
        self._lock = threading.Lock()
        # One-time OS cost: allocate and lock pages at startup.
        self._buffers: List[torch.Tensor] = [
            torch.empty(slot_bytes, dtype=torch.uint8).pin_memory()
            for _ in range(n_slots)
        ]
        self._free: List[int] = list(range(n_slots))
        logger.info(
            "PinnedMemoryPool: %d slots × %.1f MiB = %.1f MiB locked",
            n_slots,
            slot_bytes / 1024**2,
            n_slots * slot_bytes / 1024**2,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pin(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Copy *state_dict* tensors into a free pinned slot (memcpy only).

        Non-CPU tensors (e.g. already on GPU via GDS) are passed through
        unchanged.  Falls back to per-tensor ``pin_memory()`` if the pool
        slot is too small (should not happen when sized with
        :func:`compute_max_layer_bytes`).

        The returned dict holds views into the pinned buffer.  The slot is
        returned to the free list **automatically** when the dict is garbage-
        collected (via ``weakref.finalize``), so callers do not need to
        manage slot lifetimes explicitly.

        Returns:
            A new dict whose CPU tensors are backed by a pinned buffer.
        """
        slot = self._acquire_slot()
        buf = self._buffers[slot]
        offset = 0
        result: _WeakRefDict = _WeakRefDict()

        for k, t in state_dict.items():
            if t.device.type != "cpu":
                result[k] = t
                continue
            # Ensure contiguous layout before computing byte count.
            if not t.is_contiguous():
                t = t.contiguous()
            nb = t.numel() * t.element_size()
            if offset + nb > self._slot_bytes:
                # Slot too small — fall back gracefully without raising.
                logger.warning(
                    "PinnedMemoryPool slot too small (%d B needed, %d B available);"
                    " falling back to pin_memory() for layer",
                    offset + nb,
                    self._slot_bytes,
                )
                self._release_slot(slot)
                return _WeakRefDict(
                    (k2, v.pin_memory() if v.device.type == "cpu" else v)
                    for k2, v in state_dict.items()
                )
            view = buf[offset : offset + nb].view(t.dtype).reshape(t.shape)
            view.copy_(t)
            result[k] = view
            offset += nb

        # Register automatic slot release when the result dict is GC'd.
        # In CPython this fires as soon as the last reference is dropped.
        weakref.finalize(result, self._release_slot, slot)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquire_slot(self) -> int:
        """Block (spin) until a free slot is available and return its index."""
        import time as _time

        while True:
            with self._lock:
                if self._free:
                    return self._free.pop(0)
            _time.sleep(0.0005)

    def _release_slot(self, slot: int) -> None:
        with self._lock:
            self._free.append(slot)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def slot_bytes(self) -> int:
        """Byte size of each slot."""
        return self._slot_bytes

    @property
    def n_slots(self) -> int:
        """Total number of slots (free + in-use)."""
        return len(self._buffers)

    @property
    def free_slots(self) -> int:
        """Number of currently available slots."""
        with self._lock:
            return len(self._free)


# ---------------------------------------------------------------------------
# Checkpoint size helpers
# ---------------------------------------------------------------------------


def _safetensor_data_bytes(filepath: Path) -> int:
    """Return the total tensor data bytes in a safetensors file (header-only read)."""
    try:
        with open(filepath, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len).decode("utf-8"))
        total = 0
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, dict):
                continue
            offsets = meta.get("data_offsets")
            if offsets and len(offsets) == 2:
                total += offsets[1] - offsets[0]
        return total
    except Exception as exc:
        logger.debug("Could not read safetensors header %s: %s", filepath, exc)
        return 0


def compute_max_layer_bytes(checkpoint_path: str, layer_names: List[str]) -> int:
    """Scan the checkpoint and return the byte size of the largest layer shard.

    Reads only the safetensors file headers (a few hundred bytes each) — no
    tensor data is loaded.  Use the result as ``slot_bytes`` when constructing
    a :class:`PinnedMemoryPool`.

    Args:
        checkpoint_path: Path to the split-layer checkpoint directory.
        layer_names: List of layer names to scan (e.g. from ``self.layer_names``).

    Returns:
        Maximum layer shard size in bytes, or 0 if no files were found.
    """
    base = Path(checkpoint_path)
    max_bytes = 0
    for layer_name in layer_names:
        for suffix in (layer_name + "safetensors", layer_name + ".safetensors"):
            path = base / suffix
            if path.exists():
                max_bytes = max(max_bytes, _safetensor_data_bytes(path))
                break
    if max_bytes == 0:
        logger.warning(
            "compute_max_layer_bytes: no safetensors files found under %s", checkpoint_path
        )
    return max_bytes


def build_pool_for_checkpoint(
    checkpoint_path: str,
    layer_names: List[str],
    n_slots: int = 3,
    overhead_factor: float = 1.05,
    max_pool_bytes: int = 3 * 1024 ** 3,
) -> Optional[PinnedMemoryPool]:
    """Build a :class:`PinnedMemoryPool` sized for the given checkpoint.

    Scans safetensors headers to find the largest layer, adds a small overhead
    margin (default 5 %), and allocates ``n_slots`` pinned buffers.

    Returns ``None`` when CUDA is not available (pinning has no benefit on CPU),
    when the checkpoint directory contains no safetensors files, or when the
    total pool size would exceed ``max_pool_bytes``.

    The ``max_pool_bytes`` guard prevents allocating excessive pinned RAM for
    large unquantized models (e.g. 72B bfloat16 layers are ~2.5 GB each →
    3 slots = 7.5 GB locked, which starves the OS page cache and causes swap).
    The default cap is 3 GiB, which comfortably covers 4-bit layers (~700 MiB ×
    3 = 2.1 GiB) while gracefully skipping the pool for full-precision models.

    Args:
        checkpoint_path: Path to the split-layer checkpoint directory.
        layer_names: All layer names (used to locate shard files).
        n_slots: Number of concurrent slots (3 recommended for dual-prefetch).
        overhead_factor: Multiplier applied to the largest shard size to give
            a small safety margin for alignment / rounding.
        max_pool_bytes: Upper bound on total pinned pool size in bytes.  If
            ``n_slots × slot_bytes`` would exceed this value the pool is not
            created and ``None`` is returned.  Set to 0 to disable the pool
            entirely; set to a very large value to remove the cap.
    """
    try:
        from ..utils.platform import is_cuda_available

        if not is_cuda_available():
            return None
    except Exception:
        if not torch.cuda.is_available():
            return None

    max_bytes = compute_max_layer_bytes(checkpoint_path, layer_names)
    if max_bytes == 0:
        return None

    slot_bytes = int(max_bytes * overhead_factor)
    total_bytes = n_slots * slot_bytes
    if max_pool_bytes > 0 and total_bytes > max_pool_bytes:
        logger.info(
            "PinnedMemoryPool: skipped — total size %.1f MiB exceeds cap %.1f MiB"
            " (slot=%.1f MiB × %d slots). Using pin_memory() fallback.",
            total_bytes / 1024**2,
            max_pool_bytes / 1024**2,
            slot_bytes / 1024**2,
            n_slots,
        )
        return None

    try:
        return PinnedMemoryPool(n_slots=n_slots, slot_bytes=slot_bytes)
    except Exception as exc:
        logger.warning("PinnedMemoryPool allocation failed: %s — pin_memory() will be used", exc)
        return None
