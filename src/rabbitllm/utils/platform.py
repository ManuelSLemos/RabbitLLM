from sys import platform

import torch

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True


def is_macos() -> bool:
    """Return True if running on macOS (Darwin)."""
    return is_on_mac_os


def is_cuda_available() -> bool:
    """Return True if CUDA is available for PyTorch."""
    return torch.cuda.is_available()


def _flash_attn_runtime_check():
    """Run a minimal flash_attn op on the current CUDA device to verify it works.

    Catches ABI/CUDA version mismatches where the package imports but fails at runtime.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    try:
        from flash_attn import flash_attn_func

        # Minimal check: batch=1, seq=2, nheads=1, head_dim=64 (typical and supported)
        device = torch.device("cuda", torch.cuda.current_device())
        q = torch.randn(1, 2, 1, 64, dtype=torch.float16, device=device)
        k = torch.randn(1, 2, 1, 64, dtype=torch.float16, device=device)
        v = torch.randn(1, 2, 1, 64, dtype=torch.float16, device=device)
        flash_attn_func(q, k, v, causal=True)
        return True, None
    except Exception as e:
        return False, str(e)


def is_flash_attention_available():
    """Check if Flash Attention 2 is usable on this system (install + GPU + runtime).

    With ``attn_implementation="auto"`` (the default), the model uses this to decide
    whether to use Flash Attention 2 when dtype is fp16/bf16. No user action needed
    on compatible machines (Ampere+ GPU, flash-attn installed).

    Returns
    -------
    available : bool
        Whether Flash Attention 2 can be used.
    message : str
        Human-readable explanation of the result.
    """
    try:
        import flash_attn  # noqa: F401

        flash_installed = True
    except ImportError:
        flash_installed = False

    if not flash_installed:
        return (
            False,
            "flash-attn package is not installed"
            " (install with: pip install flash-attn or uv sync --extra flash)",
        )

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    device_cap = torch.cuda.get_device_capability()
    if device_cap[0] < 8:
        return False, (
            f"GPU {torch.cuda.get_device_name()} has compute capability "
            f"{device_cap[0]}.{device_cap[1]}, but Flash Attention 2 requires >= 8.0 (Ampere+)"
        )

    # Verify flash_attn actually runs on this device (catches ABI/CUDA mismatches)
    ok, err = _flash_attn_runtime_check()
    if not ok:
        return False, f"Flash Attention failed at runtime: {err}"

    return True, f"Flash Attention 2 available on {torch.cuda.get_device_name()}"
