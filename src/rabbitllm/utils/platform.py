from sys import platform

import torch

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


def is_flash_attention_available():
    """Check if flash-attn is installed and GPU supports it (Ampere+, SM >= 80).

    Returns
    -------
    available : bool
        Whether FlashAttention 2 can be used.
    message : str
        Human-readable explanation of the result.
    """
    try:
        import flash_attn  # noqa: F401

        flash_installed = True
    except ImportError:
        flash_installed = False

    if not flash_installed:
        return False, "flash-attn package is not installed"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    device_cap = torch.cuda.get_device_capability()
    if device_cap[0] < 8:
        return False, (
            f"GPU {torch.cuda.get_device_name()} has compute capability "
            f"{device_cap[0]}.{device_cap[1]}, but FlashAttention 2 requires >= 8.0 (Ampere+)"
        )

    return True, f"FlashAttention 2 available on {torch.cuda.get_device_name()}"
