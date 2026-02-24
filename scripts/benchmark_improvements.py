#!/usr/bin/env python3
"""
RabbitLLM benchmark — demonstrates GDS and DiskKVCache improvements.

Run:
  python scripts/benchmark_improvements.py --mode gds
  python scripts/benchmark_improvements.py --mode long_context

GDS (kvikio): GPU Direct Storage loads layers disk→GPU, bypassing CPU and
pin_memory. For 72B bfloat16, pin_memory was ~200s; with GDS it drops to ~6s.

DiskKVCache: Offloads the KV cache to SSD so that GPU holds at most ONE layer's
K/V at a time.  On each forward pass a single DiskKVCache object is shared across
all decoder layers; update() loads the previous K/V from disk, appends the new
token, saves back, and returns the combined tensors — then releases GPU memory.
The same cache object is returned as past_key_values so generate() passes it back
on the next decode step.  This prevents VRAM accumulation across layers and tokens,
enabling full-precision 70B+ inference on 8 GB GPUs without quantization.
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
import warnings

import torch
from rabbitllm import AutoModel
from rabbitllm.utils.kvikio_loader import kvikio_available

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _create_long_prompt(target_tokens: int = 2048) -> str:
    """Create a prompt with ~target_tokens tokens (repeating sample text)."""
    base = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    )
    sample_path = "samples/sample_2k.txt"
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            base = f.read()
    # Roughly 4 chars per token for English
    approx_chars = target_tokens * 4
    repeats = max(1, approx_chars // len(base))
    return (base * repeats)[:approx_chars]


def run_gds_benchmark(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Compare layer loading: use_gds=True vs use_gds=False (uncompressed only)."""
    if device != "cuda:0":
        print("GDS benchmark requires CUDA. Skipping.")
        return

    if not kvikio_available:
        print(
            "\n[kvikio not installed] GDS benchmark requires: pip install rabbitllm[gds]\n"
            "Installing kvikio enables disk→GPU direct load, bypassing CPU and pin_memory.\n"
            "For 72B bfloat16, reported improvement: ~200s pin_memory → ~6s with GDS.\n"
        )
        return

    print("\n=== GDS Benchmark (GPU Direct Storage) ===\n")
    print(f"Model: {model_id} (no compression, so GDS is active)")
    print("Comparing: use_gds=True vs use_gds=False")
    print()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say 'ok' in one word."},
    ]

    results = {}

    for use_gds in (False, True):
        label = "with GDS (kvikio)" if use_gds else "without GDS (CPU+pin_memory)"
        print(f"  Running {label}...")

        # Suppress profiler INFO during run
        log = logging.getLogger("rabbitllm")
        old_level = log.level
        log.setLevel(logging.WARNING)

        model = AutoModel.from_pretrained(
            model_id,
            device=device,
            compression=None,  # GDS only works without compression
            use_gds=use_gds,
            profiling_mode=True,
        )

        input_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = model.tokenizer(
            [input_text], return_tensors="pt", truncation=True, max_length=64
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        t0 = time.perf_counter()
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2,
            use_cache=True,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0
        log.setLevel(old_level)

        prof = getattr(model, "profiler", None)
        profiler_times = {}
        if prof and hasattr(prof, "profiling_time_dict"):
            for k, v in prof.profiling_time_dict.items():
                if v:
                    profiler_times[k] = sum(v)

        key = "gds" if use_gds else "no_gds"
        results[key] = {"wall_s": elapsed, "profiler_times": profiler_times}
        print(f"    Wall time: {elapsed:.2f}s")
        if profiler_times:
            load_gds = profiler_times.get("load_safe_tensor_gds", 0)
            load_cpu = profiler_times.get("load_safe_tensor", 0)
            pin_mem = profiler_times.get("pin_memory_to_trigger_load", 0)
            if use_gds and load_gds:
                print(f"    Layer load (GDS): {load_gds:.2f}s")
            elif not use_gds:
                print(f"    Layer load (disk): {load_cpu:.2f}s, pin_memory: {pin_mem:.2f}s")

    # Summary
    print()
    print("--- Results ---")
    no_gds = results.get("no_gds", {}).get("wall_s", 0)
    gds_t = results.get("gds", {}).get("wall_s", 0)
    if no_gds > 0 and gds_t > 0:
        speedup = no_gds / gds_t
        print(f"  Without GDS: {no_gds:.2f}s")
        print(f"  With GDS:    {gds_t:.2f}s")
        print(f"  Speedup:     {speedup:.2f}x")
        print()
        print("  Note: For 72B bfloat16, GDS eliminates ~200s of pin_memory;")
        print("  with 0.5B the gains are smaller but the path is exercised.")
    print()


def run_long_context_benchmark(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_tokens: int = 2048,
    max_new_tokens: int = 32,
):
    """Demonstrate DiskKVCache: long context without OOM, no quantization required.

    With kv_cache_dir set, a single DiskKVCache is created per generation run.
    Each layer's K/V is saved to disk and freed from GPU immediately after the
    attention step, so VRAM holds at most one layer's K/V at any point.
    """
    print("\n=== Long Context (DiskKVCache — no quantization) ===\n")
    print(f"Model: {model_id}")
    print(f"Context length: ~{target_tokens} tokens")
    print(f"Compression: none (full model quality)")
    print(f"KV cache: offloaded to disk (one layer at a time in VRAM)")
    print()

    kv_dir = tempfile.mkdtemp(prefix="rabbitllm_kv_")
    try:
        prompt = _create_long_prompt(target_tokens)

        model = AutoModel.from_pretrained(
            model_id,
            device=device,
            compression=None,  # full precision — no quantization required
            kv_cache_dir=kv_dir,
            max_seq_len=target_tokens + 128,
        )

        messages = [
            {"role": "system", "content": "Summarize the following text in one short sentence."},
            {"role": "user", "content": prompt},
        ]

        input_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = model.tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=target_tokens + 64,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device)

        actual_tokens = input_ids.shape[1]
        print(f"  Input tokens: {actual_tokens}")

        t0 = time.perf_counter()
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
        )
        elapsed = time.perf_counter() - t0

        reply = model.tokenizer.decode(
            output.sequences[0][actual_tokens:], skip_special_tokens=True
        )

        print(f"  Wall time: {elapsed:.2f}s")
        print(f"  Reply: {reply.strip()[:120]}...")
        print()
        print("  DiskKVCache: GPU holds at most one layer's K/V at a time.")
        print("  Full-precision 70B+ inference on 8 GB GPUs — no quantization needed.")
        print()
    finally:
        shutil.rmtree(kv_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="RabbitLLM improvement benchmarks")
    parser.add_argument(
        "--mode",
        choices=["gds", "long_context"],
        default="gds",
        help="gds: compare use_gds on/off | long_context: run with kv_cache_dir",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model ID")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=2048,
        help="Target context length for long_context mode",
    )
    args = parser.parse_args()

    if args.mode == "gds":
        run_gds_benchmark(model_id=args.model)
    else:
        run_long_context_benchmark(
            model_id=args.model, target_tokens=args.target_tokens
        )


if __name__ == "__main__":
    main()
