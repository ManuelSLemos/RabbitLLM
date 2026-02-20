"""Layer loading from disk and moving to device."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils.modeling import set_module_tensor_to_device

from ..utils import load_layer
from ..utils.platform import is_cuda_available

logger = logging.getLogger(__name__)


def load_layer_to_cpu(
    checkpoint_path: str,
    layer_name: str,
    profiling_mode: bool,
    prefetching: bool,
    profiler: Optional[Any] = None,
    persister: Optional[Any] = None,
    use_pin_memory: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load a layer's state_dict from checkpoint to CPU, optionally with pin_memory for prefetch.

    Parameters
    ----------
    checkpoint_path : str
        Path to the split checkpoint directory.
    layer_name : str
        Layer name (e.g. "model.layers.0").
    profiling_mode : bool
        Whether to record timing in profiler.
    prefetching : bool
        If True and CUDA available and use_pin_memory, pin memory for faster transfer.
    profiler : object, optional
        If provided and profiling_mode, must have add_profiling_time(name, elapsed).
    persister : object, optional
        ModelPersister for reading layer files.
    use_pin_memory : bool
        If True (default), call pin_memory() when prefetching on CUDA. Set to False for
        very large models where the cost of pin_memory dominates total time.

    Returns
    -------
    dict
        state_dict for the layer.
    """
    t = time.time()
    load_layer_output = load_layer(
        checkpoint_path, layer_name, profiling_mode, persister=persister
    )
    elapsed_time = time.time() - t

    if profiling_mode:
        state_dict, compression_time = load_layer_output
        disk_loading_time = elapsed_time - compression_time
        if profiler is not None:
            profiler.add_profiling_time("load_safe_tensor", disk_loading_time)
            profiler.add_profiling_time("compression_time", compression_time)
    else:
        state_dict = load_layer_output

    if prefetching and use_pin_memory:
        t = time.time()
        if is_cuda_available():
            for k in state_dict.keys():
                state_dict[k] = state_dict[k].pin_memory()
        else:
            logger.debug("Prefetching is enabled, but no pin_memory operation is needed for CPU.")
        elapsed_time = time.time() - t
        if profiling_mode and profiler is not None:
            profiler.add_profiling_time("pin_memory_to_trigger_load", elapsed_time)

    return state_dict


def _param_list_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    hf_quantizer: Optional[Any],
    model: Any,
) -> List[str]:
    """Return list of param names to move (one per param, or per layer for quantizer)."""
    layers = []
    for param_name in state_dict:
        if hf_quantizer is None:
            layers.append(param_name)
        else:
            if ".weight" in param_name:
                layer_name = param_name[: param_name.index(".weight") + len(".weight")]
                if layer_name not in layers:
                    layers.append(layer_name)
    return layers


def move_layer_to_device(
    model: Any,
    state_dict: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    hf_quantizer: Optional[Any] = None,
) -> List[str]:
    """Move a layer's state_dict onto the target device (or quantize and move).

    Parameters
    ----------
    model : nn.Module
        The meta model to which parameters are assigned.
    state_dict : dict
        Layer state_dict (param_name -> tensor).
    device : str
        Target device (e.g. "cuda:0").
    dtype : torch.dtype
        Target dtype.
    hf_quantizer : object, optional
        If set, used to check/create quantized params; must have
        check_quantized_param(model, param_value, param_name, state_dict),
        update_torch_dtype(device_map), create_quantized_param(model, tensor, param_name, device, state_dict).

    Returns
    -------
    list of str
        Parameter names that were moved (for later moving back to meta).
    """
    layers = _param_list_from_state_dict(state_dict, hf_quantizer, model)

    for param_name in layers:
        if hf_quantizer is None or not hf_quantizer.check_quantized_param(
            model, param_value=None, param_name=param_name, state_dict={}
        ):
            set_module_tensor_to_device(
                model,
                param_name,
                device,
                value=state_dict[param_name],
                dtype=dtype,
            )
        else:
            torch_dtype = hf_quantizer.update_torch_dtype(None)
            hf_quantizer.create_quantized_param(
                model,
                state_dict[param_name],
                param_name,
                device,
                state_dict,
            )
    return layers


def copy_layer_to_device_async(
    state_dict: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    stream: torch.cuda.Stream,
    hf_quantizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """Copy a layer's state_dict to device on a CUDA stream (no module assignment).

    Caller must synchronize the stream and then call set_layer_params_from_tensors
    on the default stream to assign the tensors to the model. Returns (tensors_dict,
    param_names) for use in set_layer_params_from_tensors and for moving back to meta.
    """
    param_names = _param_list_from_state_dict(state_dict, hf_quantizer, model)
    tensors_on_device: Dict[str, torch.Tensor] = {}
    with torch.cuda.stream(stream):
        for param_name in param_names:
            if hf_quantizer is None or not hf_quantizer.check_quantized_param(
                model, param_value=None, param_name=param_name, state_dict={}
            ):
                t = state_dict[param_name].to(device, dtype=dtype, non_blocking=True)
                tensors_on_device[param_name] = t
    return tensors_on_device, param_names


def _set_param_direct(model: Any, param_name: str, tensor: torch.Tensor) -> None:
    """Set a single parameter by full path (e.g. model.layers.0.input_layernorm.weight) without accelerate.

    Uses in-place data replacement (param.data = tensor) when the parameter already exists,
    to avoid replacing the Parameter object (which can cause stream-visibility issues when the
    tensor was produced on a non-default CUDA stream). Falls back to setattr for missing params.
    """
    parts = param_name.split(".")
    parent = model
    for i in range(len(parts) - 1):
        part = parts[i]
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    existing = parent._parameters.get(parts[-1])
    requires_grad = existing.requires_grad if existing is not None else False
    new_param = torch.nn.Parameter(tensor.detach(), requires_grad=requires_grad)
    if existing is not None and existing.device == tensor.device:
        # Same device: update data in-place (keeps Parameter object identity)
        existing.data = new_param.data
    elif existing is not None:
        # Different device (e.g. meta → cuda): replace directly in _parameters dict
        parent._parameters[parts[-1]] = new_param
    else:
        setattr(parent, parts[-1], new_param)


def set_layer_params_from_tensors(
    model: Any,
    tensors_on_device: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    use_clone_fallback: bool = False,
    use_direct_set: bool = False,
) -> List[str]:
    """Assign already-on-device tensors to the model (call on default stream after sync).

    Call this after transfer_stream.synchronize() and wait_stream() so the tensors
    are visible. If use_clone_fallback is True, clones each tensor on the default
    stream before assigning. If use_direct_set is True, sets parameters via
    getattr/setattr instead of accelerate (avoids meta/cuda issues in some envs).
    """
    param_names = list(tensors_on_device.keys())
    for param_name in param_names:
        t = tensors_on_device[param_name]
        if use_clone_fallback:
            # Force tensor to be created on default stream (empty_like + copy_ run on current stream)
            t = torch.empty_like(t, device=t.device, dtype=t.dtype).copy_(t)
        if use_direct_set:
            _set_param_direct(model, param_name, t)
        else:
            set_module_tensor_to_device(
                model,
                param_name,
                device,
                value=t,
                dtype=dtype,
            )
    return param_names


def move_layer_to_device_async(
    model: Any,
    state_dict: Dict[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    stream: Optional[torch.cuda.Stream] = None,
    hf_quantizer: Optional[Any] = None,
) -> Union[List[str], Tuple[Dict[str, torch.Tensor], List[str]]]:
    """Copy a layer's state_dict to device on a CUDA stream (async path) or move synchronously.

    When stream is set and device is CUDA: only copies to GPU on the transfer stream;
    returns (tensors_dict, param_names). Caller must sync, wait_stream, then call
    set_layer_params_from_tensors(model, tensors_dict, device, dtype) on the default stream.
    When stream is None or device is CPU or hf_quantizer is set: falls back to
    move_layer_to_device and returns List[str].
    """
    if stream is None or not device.startswith("cuda") or hf_quantizer is not None:
        return move_layer_to_device(model, state_dict, device, dtype, hf_quantizer)

    tensors_on_device, param_names = copy_layer_to_device_async(
        state_dict, device, dtype, stream, hf_quantizer, model
    )
    return tensors_on_device, param_names
