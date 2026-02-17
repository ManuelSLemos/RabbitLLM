import gc
import ctypes

import torch


class NotEnoughSpaceException(Exception):
    pass


def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    torch.cuda.empty_cache()
