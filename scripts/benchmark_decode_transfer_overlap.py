#!/usr/bin/env python3
"""Benchmark: CPU→GPU transfer overlap in decode vs prefill.

Measures the actual PCIe transfer time for a layer-sized tensor, then
simulates Phase B stall by comparing transfer duration against the
forward time for prefill vs decode scenarios.

Usage:
    uv run python scripts/benchmark_decode_transfer_overlap.py
    uv run python scripts/benchmark_decode_transfer_overlap.py --layer-mb 700 --n-iters 5
"""

import argparse
import statistics
import time

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pinned(total_mb: float, n_tensors: int = 7) -> list[torch.Tensor]:
    """Create pinned CPU tensors summing to total_mb."""
    each_elems = int(total_mb * 1024 * 1024 / n_tensors / 2)  # bfloat16 → 2 bytes
    return [torch.empty(each_elems, dtype=torch.bfloat16).pin_memory() for _ in range(n_tensors)]


def _gpu_sink(device: str) -> torch.Tensor:
    """Pre-allocate GPU buffer to absorb copies (avoid first-alloc cost)."""
    return torch.empty(1, device=device)


# ---------------------------------------------------------------------------
# Transfer time measurement
# ---------------------------------------------------------------------------


def measure_transfer_time_ms(pinned_tensors: list[torch.Tensor], device: str, n_reps: int = 5) -> float:
    """Measure synchronous CPU→GPU transfer time for a set of pinned tensors."""
    stream = torch.cuda.Stream(device=device)
    times = []
    for _ in range(n_reps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            gpu = [t.to(device, non_blocking=True) for t in pinned_tensors]
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        del gpu
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Phase B stall simulation
# ---------------------------------------------------------------------------


def simulate_phase_b_stall(
    pinned_tensors: list[torch.Tensor],
    forward_ms: float,
    device: str,
    n_layers: int = 8,
) -> list[float]:
    """Simulate async-transfer pipeline Phase B stall per layer.

    Each iteration:
      1. (Phase A) Launch async CPU→GPU copy on transfer_stream
      2. Simulate GPU forward of `forward_ms` on default stream
      3. (Phase B) Call transfer_stream.synchronize() — measure the stall

    Returns list of stall times in ms (one per layer).
    """
    transfer_stream = torch.cuda.Stream(device=device)
    stalls = []

    # Prime the pipeline: kick off copy for layer 0 before the loop
    with torch.cuda.stream(transfer_stream):
        pending = [t.to(device, non_blocking=True) for t in pinned_tensors]

    for _ in range(n_layers):
        # Phase A: start async copy for next layer (reuse same pinned tensors for simplicity)
        with torch.cuda.stream(transfer_stream):
            next_gpu = [t.to(device, non_blocking=True) for t in pinned_tensors]

        # Simulate forward: burn `forward_ms` of real wall-clock time on a real GPU kernel
        if forward_ms > 0:
            # Large enough matmul to consume real GPU time
            n = 4096
            a = torch.randn(n, n, device=device, dtype=torch.float16)
            b = torch.randn(n, n, device=device, dtype=torch.float16)
            reps = max(1, int(forward_ms / 2))  # rough scaling
            for _ in range(reps):
                a = torch.mm(a, b)
            torch.cuda.synchronize(device)
            del a, b

        # Phase B: how long do we stall waiting for the copy kicked off LAST iteration?
        t0 = time.perf_counter()
        transfer_stream.synchronize()
        stalls.append((time.perf_counter() - t0) * 1000.0)

        del pending
        pending = next_gpu

    del pending
    return stalls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Measure Phase B stall: decode vs prefill")
    parser.add_argument("--layer-mb", type=float, default=700.0,
                        help="Simulated layer size in MiB (default 700)")
    parser.add_argument("--n-layers", type=int, default=8,
                        help="Layers to simulate (default 8)")
    parser.add_argument("--n-iters", type=int, default=3,
                        help="Repetitions per scenario for averaging (default 3)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    print(f"\nBenchmark: CPU→GPU transfer overlap  "
          f"(layer={args.layer_mb:.0f} MiB, device={device})")
    print("=" * 70)

    pinned = _make_pinned(args.layer_mb)
    actual_mb = sum(t.numel() * t.element_size() for t in pinned) / 1024 / 1024
    _gpu_sink(device)  # warm up GPU allocator

    # --- Measure actual transfer time ---
    print(f"\n  Measuring PCIe transfer time for {actual_mb:.1f} MiB ({len(pinned)} tensors)...")
    transfer_ms = measure_transfer_time_ms(pinned, device, n_reps=5)
    pcie_gb_s = actual_mb / 1024 / (transfer_ms / 1000)
    print(f"  → Transfer time: {transfer_ms:.1f} ms  ({pcie_gb_s:.1f} GB/s effective)")

    # Calibrate forward durations: measure what the mm loop actually produces
    scenarios = [
        ("prefill", 50),   # 50 matmul reps ≈ long forward
        ("decode",   1),   # 1 matmul rep  ≈ single-token forward
    ]

    print()
    for name, reps in scenarios:
        all_stalls = []
        all_fwd_ms = []
        for _ in range(args.n_iters):
            # Time the forward kernel separately
            n = 4096
            a = torch.randn(n, n, device=device, dtype=torch.float16)
            b = torch.randn(n, n, device=device, dtype=torch.float16)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for _ in range(reps):
                a = torch.mm(a, b)
            torch.cuda.synchronize(device)
            fwd_ms = (time.perf_counter() - t0) * 1000.0
            all_fwd_ms.append(fwd_ms)
            del a, b

            # Simulate pipeline stall
            stalls = simulate_phase_b_stall(
                pinned, forward_ms=0,  # we timed separately above; 0 skips re-running
                device=device, n_layers=args.n_layers
            )
            all_stalls.extend(stalls)

        mean_fwd = statistics.mean(all_fwd_ms)
        mean_stall = statistics.mean(all_stalls)
        max_stall = max(all_stalls)
        total_stall = mean_stall * args.n_layers

        print(f"  [{name.upper()}]")
        print(f"    Actual forward/layer: {mean_fwd:.1f} ms  (via {reps}× 4096² fp16 matmul)")
        print(f"    Phase B sync stall:   mean={mean_stall:.1f}ms  max={max_stall:.1f}ms  "
              f"total={total_stall:.0f}ms over {args.n_layers} layers")
        overlap = max(0, transfer_ms - mean_fwd)
        print(f"    Predicted stall (transfer - forward): "
              f"{overlap:.1f}ms = {transfer_ms:.1f} - {mean_fwd:.1f}")
        print()

    print(f"  Transfer time ({transfer_ms:.1f}ms) vs decode forward ({scenarios[1][1]} matmul reps):")
    print(f"  → If transfer >> forward, Phase B stalls; lookahead +1 eliminates it.")
    print()


if __name__ == "__main__":
    main()
