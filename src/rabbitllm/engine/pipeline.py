"""Layer streaming pipeline strategies.

Each pipeline generator manages the disk→CPU→GPU transfer order and yields
``(layer_name, state_dict, moved_layers)`` for every layer in sequence.
The caller (base.py) runs the forward pass after each yield, moves the layer
back to meta if needed, then advances the generator.  Cleanup (freeing
``state_dict`` and flushing allocators) happens at the start of the next
``next()`` call so it always runs after the layer is on meta.

Usage::

    pipeline = create_pipeline(...)
    for layer_name, state_dict, moved_layers in pipeline:
        # run forward for this layer
        # move layer back to meta
        # advance iterator (for-loop does this automatically)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, FrozenSet, Generator, List, Optional, Tuple

import torch

from . import layer_loading as ll
from ..utils.memory import clean_memory

logger = logging.getLogger(__name__)

# Type yielded by every pipeline variant.
PipelineItem = Tuple[str, Dict[str, Any], List[str]]


def create_pipeline(
    *,
    layer_names: List[str],
    layers: List[Any],
    load_fn: Callable[[str], Dict],
    model: Any,
    device: str,
    dtype: torch.dtype,
    hf_quantizer: Optional[Any],
    transfer_stream: Optional[Any],
    transfer_stream_b: Optional[Any] = None,
    prefetching: bool,
    use_async_transfer: bool,
    use_dual_prefetch: bool,
    use_decode_extra_lookahead: bool = False,
    async_decompress: bool,
    small_layer_names: Tuple[str, ...],
    small_layers_on_gpu: bool,
    profiling_mode: bool,
    profiler: Optional[Any],
) -> Generator[PipelineItem, None, None]:
    """Return the appropriate pipeline generator for the given hardware config."""
    common = dict(
        layer_names=layer_names,
        layers=layers,
        load_fn=load_fn,
        model=model,
        device=device,
        dtype=dtype,
        hf_quantizer=hf_quantizer,
        small_layer_names=small_layer_names,
        small_layers_on_gpu=small_layers_on_gpu,
        profiling_mode=profiling_mode,
        profiler=profiler,
    )
    if use_async_transfer:
        return _async_transfer_pipeline(
            **common,
            transfer_stream=transfer_stream,
            transfer_stream_b=transfer_stream_b,
            use_dual_prefetch=use_dual_prefetch,
            use_decode_extra_lookahead=use_decode_extra_lookahead,
            async_decompress=async_decompress,
        )
    if prefetching:
        return _sync_prefetch_pipeline(**common)
    return _no_prefetch_pipeline(**common)


# ---------------------------------------------------------------------------
# No-prefetch pipeline
# ---------------------------------------------------------------------------

def _no_prefetch_pipeline(
    *,
    layer_names: List[str],
    layers: List[Any],
    load_fn: Callable[[str], Dict],
    model: Any,
    device: str,
    dtype: torch.dtype,
    hf_quantizer: Optional[Any],
    small_layer_names: Tuple[str, ...],
    small_layers_on_gpu: bool,
    profiling_mode: bool,
    profiler: Optional[Any],
) -> Generator[PipelineItem, None, None]:
    """Load each layer synchronously from disk, then move to device."""
    prev_state_dict: Optional[Dict] = None

    for layer_name, _layer in zip(layer_names, layers):
        if prev_state_dict is not None:
            prev_state_dict = None
            clean_memory()

        if layer_name in small_layer_names and small_layers_on_gpu:
            state_dict: Dict = {}
            moved_layers: List[str] = []
        else:
            if profiling_mode:
                t = time.time()
            state_dict = load_fn(layer_name)
            if profiling_mode and profiler is not None:
                profiler.add_profiling_time("load_safe_tensor_cpu_wait", time.time() - t)
                t = time.time()
            moved_layers = ll.move_layer_to_device(model, state_dict, device, dtype, hf_quantizer)
            if profiling_mode and profiler is not None:
                profiler.add_profiling_time("create_layer_from_safe_tensor", time.time() - t)

        prev_state_dict = state_dict
        yield layer_name, state_dict, moved_layers

    if prev_state_dict is not None:
        prev_state_dict = None
        clean_memory()


# ---------------------------------------------------------------------------
# Sync-prefetch pipeline (background CPU load, sync GPU move)
# ---------------------------------------------------------------------------

def _sync_prefetch_pipeline(
    *,
    layer_names: List[str],
    layers: List[Any],
    load_fn: Callable[[str], Dict],
    model: Any,
    device: str,
    dtype: torch.dtype,
    hf_quantizer: Optional[Any],
    small_layer_names: Tuple[str, ...],
    small_layers_on_gpu: bool,
    profiling_mode: bool,
    profiler: Optional[Any],
) -> Generator[PipelineItem, None, None]:
    """Overlap CPU loading of layer i+1 with GPU forward pass of layer i."""
    prev_state_dict: Optional[Dict] = None

    with ThreadPoolExecutor() as executor:
        if layer_names:
            if profiling_mode:
                t = time.time()
            future = executor.submit(load_fn, layer_names[0])

        for i, (layer_name, _layer) in enumerate(zip(layer_names, layers)):
            if prev_state_dict is not None:
                prev_state_dict = None
                clean_memory()

            if layer_name in small_layer_names and small_layers_on_gpu:
                state_dict: Dict = {}
                moved_layers: List[str] = []
                if (i + 1) < len(layer_names):
                    future = executor.submit(load_fn, layer_names[i + 1])
            else:
                if profiling_mode and profiler is not None:
                    t = time.time()
                state_dict = future.result()
                if profiling_mode and profiler is not None:
                    profiler.add_profiling_time("load_safe_tensor_cpu_wait", time.time() - t)
                    t = time.time()
                moved_layers = ll.move_layer_to_device(
                    model, state_dict, device, dtype, hf_quantizer
                )
                if profiling_mode and profiler is not None:
                    profiler.add_profiling_time("create_layer_from_state_dict", time.time() - t)
                    t = time.time()
                if (i + 1) < len(layer_names):
                    future = executor.submit(load_fn, layer_names[i + 1])
                if profiling_mode and profiler is not None:
                    profiler.add_profiling_time("kick_off_load_cpu", time.time() - t)

            prev_state_dict = state_dict
            yield layer_name, state_dict, moved_layers

    if prev_state_dict is not None:
        prev_state_dict = None
        clean_memory()


# ---------------------------------------------------------------------------
# Async-transfer pipeline (Phase A/B: CPU load + GPU copy on transfer_stream)
# ---------------------------------------------------------------------------

def _async_transfer_pipeline(
    *,
    layer_names: List[str],
    layers: List[Any],
    load_fn: Callable[[str], Dict],
    model: Any,
    device: str,
    dtype: torch.dtype,
    hf_quantizer: Optional[Any],
    transfer_stream: Any,
    transfer_stream_b: Optional[Any] = None,
    use_dual_prefetch: bool,
    use_decode_extra_lookahead: bool = False,
    async_decompress: bool,
    small_layer_names: Tuple[str, ...],
    small_layers_on_gpu: bool,
    profiling_mode: bool,
    profiler: Optional[Any],
) -> Generator[PipelineItem, None, None]:
    """Dual-async pipeline: overlap CPU load, CPU→GPU copy and GPU forward.

    Phase A (per iteration): kick off async GPU copy for layer i+2 on transfer_stream.
    Phase B (per iteration): after forward of layer i, sync transfer_stream and assign
    parameters for layer i+1.

    When ``use_dual_prefetch`` is True, two CPU-load slots run concurrently to
    keep both the background-load thread and the async GPU copy thread busy.

    When ``use_decode_extra_lookahead`` is True (decode mode with a second transfer
    stream available), Phase A also kicks off an additional async copy for layer i+3
    on ``transfer_stream_b``.  This gives the i+3 copy two forward-pass intervals to
    run on the PCIe bus before its Phase B synchronisation, reducing the stall when
    the transfer time is close to 2× the forward time (e.g. compressed layers on fast
    PCIe hardware).  For every other layer the benefit propagates forward through the
    pipeline so that alternating layers can be served with lower stall.
    """
    _extra = use_decode_extra_lookahead and transfer_stream_b is not None
    n_layers = len(layer_names)
    if n_layers == 0:
        return

    _async_skip_layer0 = small_layers_on_gpu  # embed is already on GPU during decode steps

    with ThreadPoolExecutor() as executor:
        if profiling_mode:
            t = time.time()
        # Extra-lookahead state: a second async copy in-flight on transfer_stream_b.
        # _pending_sync_stream tracks which stream the current _pending_tensors copy
        # is on, since after promotion the near copy may be on transfer_stream_b.
        _pending_far_tensors: Optional[Dict] = None
        _pending_far_param_names: Optional[List[str]] = None
        _pending_far_s_cpu: Optional[Dict] = None
        _pending_sync_stream = transfer_stream  # updated when far is promoted to near

        # --- Startup: prime the CPU-load futures and kick off the first async GPU copy ---
        #
        # IMPORTANT: null startup Future variables (f0, f1, fa) immediately after extracting
        # their results.  concurrent.futures.Future stores the result in _result and never
        # clears it, so a named Future variable keeps the pinned-pool slot alive for the
        # entire generator lifetime — permanently occupying slots and causing deadlock.
        if use_dual_prefetch:
            if _async_skip_layer0:
                fa = executor.submit(load_fn, layer_names[1]) if n_layers > 1 else None
                fb = executor.submit(load_fn, layer_names[2]) if n_layers > 2 else None
                fc = executor.submit(load_fn, layer_names[3]) if n_layers > 3 else None
                s0: Dict = {}
                s1 = fa.result() if fa is not None else None
                fa = None  # drop Future so its _result ref to the pinned slot is freed
                _next_cpu_future_0 = fb
                _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                _next_cpu_future_1 = fc
                _next_cpu_idx_1 = 3 if n_layers > 3 else -1
            else:
                f0 = executor.submit(load_fn, layer_names[0])
                f1 = executor.submit(load_fn, layer_names[1]) if n_layers > 1 else None
                f2 = executor.submit(load_fn, layer_names[2]) if n_layers > 2 else None
                f3 = executor.submit(load_fn, layer_names[3]) if n_layers > 3 else None
                s0 = f0.result()
                f0 = None  # drop Future _result ref
                s1 = f1.result() if f1 is not None else None
                f1 = None  # drop Future _result ref
                _next_cpu_future_0 = f2
                _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                _next_cpu_future_1 = f3
                _next_cpu_idx_1 = 3 if n_layers > 3 else -1
        else:
            if _async_skip_layer0:
                fa = executor.submit(load_fn, layer_names[1]) if n_layers > 1 else None
                fb = executor.submit(load_fn, layer_names[2]) if n_layers > 2 else None
                s0 = {}
                s1 = fa.result() if fa is not None else None
                fa = None  # drop Future _result ref
                _next_cpu_future_0 = fb
                _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                _next_cpu_future_1 = None
                _next_cpu_idx_1 = -1
            else:
                f0 = executor.submit(load_fn, layer_names[0])
                f1 = executor.submit(load_fn, layer_names[1]) if n_layers > 1 else None
                s0 = f0.result()
                f0 = None  # drop Future _result ref
                s1 = f1.result() if f1 is not None else None
                f1 = None  # drop Future _result ref
                _next_cpu_future_0 = (
                    executor.submit(load_fn, layer_names[2]) if n_layers > 2 else None
                )
                _next_cpu_idx_0 = 2 if n_layers > 2 else -1
                _next_cpu_future_1 = None
                _next_cpu_idx_1 = -1

        if profiling_mode and profiler is not None:
            profiler.add_profiling_time("load_safe_tensor_cpu_wait", time.time() - t)
            t = time.time()

        # Move layer 0 to GPU (synchronously)
        if _async_skip_layer0:
            current_moved_layers: List[str] = []
            current_s: Dict = {}
        else:
            if async_decompress and s0:
                from ..utils.compression import uncompress_layer_state_dict
                s0 = uncompress_layer_state_dict(s0)
            current_moved_layers = ll.move_layer_to_device(model, s0, device, dtype, hf_quantizer)
            current_s = s0

        if profiling_mode and profiler is not None:
            profiler.add_profiling_time("create_layer_from_state_dict", time.time() - t)

        # Kick off async GPU copy for layer 1 (runs while layer 0 forward executes)
        if s1 is not None:
            _async_result = ll.move_layer_to_device_async(
                model, s1, device, dtype, stream=transfer_stream, hf_quantizer=hf_quantizer
            )
            _pending_tensors, _pending_param_names = _async_result
            _pending_s_cpu = s1
        else:
            _pending_tensors = _pending_param_names = _pending_s_cpu = None

        s0 = s1 = None  # release initial refs so GDS buffers can be reclaimed

        # Extra-lookahead startup: consume the CPU future for layer 2 early and kick
        # off its GPU copy on transfer_stream_b.  This gives the copy two forward-pass
        # intervals before Phase B of iteration 1 needs it.
        #
        # At this point _next_cpu_future_0 holds the future for layer 2 and
        # _next_cpu_idx_0 == 2 (the use_dual_prefetch + _async_skip_layer0 startup
        # always sets it up this way, and _extra requires both flags).
        if _extra and _next_cpu_future_0 is not None and _next_cpu_idx_0 == 2:
            s2_extra = _next_cpu_future_0.result()
            _next_cpu_future_0 = None
            _far_startup = ll.move_layer_to_device_async(
                model, s2_extra, device, dtype,
                stream=transfer_stream_b, hf_quantizer=hf_quantizer,
            )
            _pending_far_tensors, _pending_far_param_names = _far_startup
            _pending_far_s_cpu = s2_extra
            # Shift futures: layer 3 (was in _next_cpu_idx_1) becomes the new _idx_0,
            # and we submit a fresh future for layer 4 into _idx_1.
            _next_cpu_future_0 = _next_cpu_future_1
            _next_cpu_idx_0 = _next_cpu_idx_1
            _next_cpu_future_1 = (
                executor.submit(load_fn, layer_names[4]) if n_layers > 4 else None
            )
            _next_cpu_idx_1 = 4 if n_layers > 4 else -1

        clean_memory()

        # --- Main iteration loop ---
        for i, (layer_name, _layer) in enumerate(zip(layer_names, layers)):
            state_dict = current_s
            moved_layers = current_moved_layers

            yield layer_name, state_dict, moved_layers

            # Consumer has finished forward + meta move — now free and advance the pipeline
            current_s = None
            clean_memory()

            # Phase B: finalize the async GPU copy that ran during this forward pass.
            # _pending_sync_stream is transfer_stream normally, or transfer_stream_b when
            # the current pending was promoted from the far slot in the previous Phase B.
            if _pending_tensors is not None:
                if profiling_mode:
                    _t_sync = time.time()
                _pending_sync_stream.synchronize()
                if profiling_mode and profiler is not None:
                    profiler.add_profiling_time("transfer_stream_sync_wait", time.time() - _t_sync)
                torch.cuda.current_stream().wait_stream(_pending_sync_stream)
                if async_decompress:
                    _pending_tensors, _pending_param_names = ll.decompress_layer_on_device(
                        _pending_tensors
                    )
                ll.set_layer_params_from_tensors(
                    model,
                    _pending_tensors,
                    device,
                    dtype,
                    use_clone_fallback=False,
                    use_direct_set=True,
                )
                current_moved_layers = _pending_param_names
                current_s = _pending_s_cpu
                _pending_tensors = _pending_param_names = _pending_s_cpu = None

                # Extra-lookahead: promote the far copy (on transfer_stream_b) to the
                # pending near slot so Phase B of the NEXT iteration syncs stream_b.
                # The far copy has been running for the entire duration of this Phase B
                # (including any stall time), giving it extra PCIe time to complete.
                if _extra and _pending_far_tensors is not None:
                    _pending_tensors = _pending_far_tensors
                    _pending_param_names = _pending_far_param_names
                    _pending_s_cpu = _pending_far_s_cpu
                    _pending_sync_stream = transfer_stream_b
                    _pending_far_tensors = _pending_far_param_names = _pending_far_s_cpu = None

            # Phase A: start async GPU copy for layer i+2 (overlaps with layer i+1 forward)
            if (i + 2) < n_layers:
                need_idx = i + 2
                if use_dual_prefetch:
                    if _next_cpu_idx_0 == need_idx and _next_cpu_future_0 is not None:
                        _next_cpu_s = _next_cpu_future_0.result()
                        _consumed_slot = 0
                    elif _next_cpu_idx_1 == need_idx and _next_cpu_future_1 is not None:
                        _next_cpu_s = _next_cpu_future_1.result()
                        _consumed_slot = 1
                    else:
                        _next_cpu_s = None
                        _consumed_slot = -1

                    if _next_cpu_s is not None:
                        _async_result = ll.move_layer_to_device_async(
                            model,
                            _next_cpu_s,
                            device,
                            dtype,
                            stream=transfer_stream,
                            hf_quantizer=hf_quantizer,
                        )
                        _pending_tensors, _pending_param_names = _async_result
                        _pending_s_cpu = _next_cpu_s
                        _pending_sync_stream = transfer_stream  # new near is on stream_a

                    if _consumed_slot == 0:
                        _next_cpu_future_0 = (
                            executor.submit(load_fn, layer_names[i + 4])
                            if (i + 4) < n_layers
                            else None
                        )
                        _next_cpu_idx_0 = (i + 4) if (i + 4) < n_layers else -1
                    elif _consumed_slot == 1:
                        _next_cpu_future_1 = (
                            executor.submit(load_fn, layer_names[i + 4])
                            if (i + 4) < n_layers
                            else None
                        )
                        _next_cpu_idx_1 = (i + 4) if (i + 4) < n_layers else -1
                else:
                    _next_cpu_s = (
                        _next_cpu_future_0.result() if _next_cpu_future_0 is not None else None
                    )
                    if _next_cpu_s is not None:
                        _async_result = ll.move_layer_to_device_async(
                            model,
                            _next_cpu_s,
                            device,
                            dtype,
                            stream=transfer_stream,
                            hf_quantizer=hf_quantizer,
                        )
                        _pending_tensors, _pending_param_names = _async_result
                        _pending_s_cpu = _next_cpu_s
                        _pending_sync_stream = transfer_stream  # new near is on stream_a
                    else:
                        _pending_tensors = _pending_param_names = _pending_s_cpu = None
                    _next_cpu_future_0 = (
                        executor.submit(load_fn, layer_names[i + 3])
                        if (i + 3) < n_layers
                        else None
                    )
                    _next_cpu_idx_0 = (i + 3) if (i + 3) < n_layers else -1
            else:
                _pending_tensors = _pending_param_names = _pending_s_cpu = None

            # Extra-lookahead Phase A: kick off an additional copy for layer i+3 on
            # transfer_stream_b.  The normal Phase A above consumed the future for i+2;
            # the remaining future (_next_cpu_idx_* == i+3) is consumed here and the
            # slot is refilled with i+5 (normal Phase A refills its slot with i+4).
            # This copy benefits from the Phase B stall of the CURRENT iteration (stream_b
            # runs while stream_a's synchronize() blocks the host), so by Phase B of the
            # NEXT iteration it has had stall_time + forward_time extra to complete.
            if _extra and (i + 3) < n_layers:
                _far_idx = i + 3
                _far_cpu: Optional[Dict] = None
                _far_consumed_slot = -1
                if _next_cpu_idx_0 == _far_idx and _next_cpu_future_0 is not None:
                    _far_cpu = _next_cpu_future_0.result()
                    _far_consumed_slot = 0
                    _next_cpu_future_0 = None
                elif _next_cpu_idx_1 == _far_idx and _next_cpu_future_1 is not None:
                    _far_cpu = _next_cpu_future_1.result()
                    _far_consumed_slot = 1
                    _next_cpu_future_1 = None
                if _far_cpu is not None:
                    _far_async = ll.move_layer_to_device_async(
                        model, _far_cpu, device, dtype,
                        stream=transfer_stream_b, hf_quantizer=hf_quantizer,
                    )
                    _pending_far_tensors, _pending_far_param_names = _far_async
                    _pending_far_s_cpu = _far_cpu
                    # Refill consumed slot with layer i+5
                    if _far_consumed_slot == 0:
                        _next_cpu_future_0 = (
                            executor.submit(load_fn, layer_names[i + 5])
                            if (i + 5) < n_layers else None
                        )
                        _next_cpu_idx_0 = (i + 5) if (i + 5) < n_layers else -1
                    else:
                        _next_cpu_future_1 = (
                            executor.submit(load_fn, layer_names[i + 5])
                            if (i + 5) < n_layers else None
                        )
                        _next_cpu_idx_1 = (i + 5) if (i + 5) < n_layers else -1
                else:
                    _pending_far_tensors = _pending_far_param_names = _pending_far_s_cpu = None

            # After Phase A, the new near copy (if any) is always on transfer_stream.
            # Reset the sync stream so Phase B of the next-but-one iteration syncs it,
            # unless Phase B already promoted a far copy this iteration (in which case
            # _pending_sync_stream is already set to transfer_stream_b).
            if _extra and _pending_sync_stream is transfer_stream_b and _pending_tensors is None:
                # No near copy was placed (i+2 out of range); no pending to sync next time.
                _pending_sync_stream = transfer_stream
