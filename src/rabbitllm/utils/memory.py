import ctypes
import gc

import torch

try:
    import cupy as cp  # type: ignore
    _cupy_available = True
except ImportError:
    cp = None  # type: ignore
    _cupy_available = False


class NotEnoughSpaceException(Exception):
    pass


def clean_memory():
    """Run gc, malloc_trim (Linux), torch.cuda.empty_cache(), and CuPy pool flush.

    When kvikio/GDS is used, tensors are backed by CuPy-owned VRAM. PyTorch's
    empty_cache() does not touch CuPy's allocator, so the pool accumulates across
    layer loads and fills the GPU. Flushing the CuPy pool returns those blocks to
    CUDA immediately, preventing OOM on low-VRAM GPUs with large models.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()
    if _cupy_available and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
