#!/usr/bin/env python3
"""
Benchmark CPU vs CUDA inference with the same model and prompt.

In layer-streaming mode, each layer is loaded from disk and moved to the device
every forward pass. For small models (e.g. 0.5B), CPU→GPU transfer overhead can
dominate, making CPU faster than CUDA. This script measures total time and
time per token for both devices so you can compare.

Usage:
    uv run python scripts/benchmark_cpu_vs_cuda.py
    uv run python scripts/benchmark_cpu_vs_cuda.py --model Qwen/Qwen2.5-1.5B-Instruct --runs 3
"""

import argparse
import time

import torch
from rabbitllm import AutoModel


def run_benchmark(device: str, model_id: str, prompt: str, max_new_tokens: int, runs: int):
    """Load model on device, run generation `runs` times; return (total_sec, n_tokens)."""
    print(f"Loading model on {device!r}...")
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, device=device)
    load_sec = time.perf_counter() - load_start
    print(f"  Loaded in {load_sec:.1f}s")

    tokenizer = model.tokenizer
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=model.device)
    else:
        attention_mask = attention_mask.to(model.device)

    # Warmup
    model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        return_dict_in_generate=True,
    )

    # Timed runs
    times = []
    for i in range(runs):
        start = time.perf_counter()
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
        elapsed = time.perf_counter() - start
        n_new = out.sequences.shape[1] - input_ids.shape[1]
        times.append((elapsed, n_new))
        print(f"  Run {i + 1}/{runs}: {elapsed:.2f}s, {n_new} new tokens")

    total_sec = sum(t for t, _ in times)
    total_tokens = sum(n for _, n in times)
    avg_sec = total_sec / runs
    avg_tokens = total_tokens / runs
    sec_per_token = total_sec / total_tokens if total_tokens else 0
    return {
        "device": device,
        "load_sec": load_sec,
        "runs": runs,
        "total_sec": total_sec,
        "total_tokens": total_tokens,
        "avg_sec_per_run": avg_sec,
        "avg_tokens_per_run": avg_tokens,
        "sec_per_token": sec_per_token,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs CUDA inference (layer-streaming).")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model id (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2+2? Answer briefly.",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max new tokens per run (default: 50)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of timed runs per device (default: 2)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Only run CPU benchmark (skip CUDA)",
    )
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Only run CUDA benchmark (skip CPU)",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"max_new_tokens: {args.max_new_tokens}, runs: {args.runs}\n")

    results = []

    if not args.cuda_only:
        results.append(
            run_benchmark("cpu", args.model, args.prompt, args.max_new_tokens, args.runs)
        )
        print()

    if not args.cpu_only:
        results.append(
            run_benchmark("cuda:0", args.model, args.prompt, args.max_new_tokens, args.runs)
        )
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['device']:10}  total={r['total_sec']:.2f}s  "
              f"avg_run={r['avg_sec_per_run']:.2f}s  "
              f"sec/token={r['sec_per_token']:.4f}  "
              f"tokens={r['total_tokens']:.0f}")
    if len(results) == 2:
        a, b = results[0], results[1]
        faster = a["device"] if a["total_sec"] < b["total_sec"] else b["device"]
        ratio = max(a["total_sec"], b["total_sec"]) / max(min(a["total_sec"], b["total_sec"]), 1e-9)
        print(f"\n  Faster: {faster} (by {ratio:.2f}x)")
    print()


if __name__ == "__main__":
    main()
