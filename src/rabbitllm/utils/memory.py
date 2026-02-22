import ctypes
import gc

import torch


class NotEnoughSpaceException(Exception):
    pass


def clean_memory():
    """Run gc, malloc_trim (Linux), and torch.cuda.empty_cache()."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()
