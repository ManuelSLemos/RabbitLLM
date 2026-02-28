import argparse
import logging
import sys
import time
import warnings

import torch
from rabbitllm import AutoModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# CLI — allows switching model / compression from the command line without
# editing the script.  Sensible defaults for a 32 GB / 8 GB VRAM laptop.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="RabbitLLM inference example")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="HuggingFace repo ID or local path")
parser.add_argument("--compression", default="4bit", choices=["4bit", "8bit", "none"],
                    help="Weight compression. Default: 4bit (recommended for 72B on ≤32 GB RAM). "
                         "bfloat16 (none) needs ~195 s/token and risks thermal shutdown on laptops.")
parser.add_argument("--max-new-tokens", type=int, default=50,
                    help="Tokens to generate. Keep small (≤10) for quick tests on 72B. "
                         "Default: 50 (~10 min with 4-bit, ~33 min with bfloat16).")
parser.add_argument("--cache-layers", type=int, default=None,
                    help="CPU RAM layer cache size (number of layers). "
                         "Speeds up decode by skipping repeated disk reads. "
                         "Suggested: 30 for 4-bit 72B on 32 GB RAM.")
parser.add_argument("--prompt", default="What is the capital of France?",
                    help="User message to send to the model.")
parser.add_argument("--kv-cache-dir", default=None, metavar="DIR",
                    help="Directory to offload the KV cache to disk. "
                         "Prevents VRAM accumulation on long contexts — use this to avoid OOM on "
                         "8 GB GPUs with full-precision 70B+ models. Each decode step loads only "
                         "the current layer's K/V from disk and frees it immediately after. "
                         "When set, also enables offload of embed/norm/lm_head to avoid OOM. "
                         "Example: --kv-cache-dir /tmp/kv_cache")
parser.add_argument("--use-gds", action="store_true",
                    help="Enable GPU Direct Storage (kvikio) for layer weight loading. "
                         "Loads safetensors directly from disk to GPU, bypassing CPU RAM. "
                         "Requires: pip install rabbitllm[gds] and compression=none. "
                         "For 72B bfloat16 this cuts weight-load time from ~200s to ~6s. "
                         "Pairs well with --kv-cache-dir: GDS speeds up weight loads, "
                         "kv-cache-dir prevents K/V VRAM accumulation.")
parser.add_argument("--offload-small-layers", action="store_true",
                    help="Offload embed/norm/lm_head to CPU each step (avoids OOM on 8 GB with 72B "
                         "without quantization). Use with --kv-cache-dir. Keeps small layers in "
                         "CPU cache by default for oLLM-like decode speed.")
parser.add_argument("--no-offload-small-layers-cpu-cache", action="store_true",
                    help="Disable CPU cache for offloaded small layers (use with --offload-small-"
                         "layers). Slower decode but lower RAM.")
parser.add_argument("--no-think", action="store_true",
                    help="Disable Qwen3 chain-of-thought thinking mode (adds /no_think system prompt).")
parser.add_argument("--do-sample", action="store_true",
                    help="Use sampling instead of greedy decoding. Recommended for thinking models "
                         "to avoid repetition loops.")
parser.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (only used with --do-sample). Default: 0.6.")
parser.add_argument("--top-p", type=float, default=0.95,
                    help="Top-p nucleus sampling (only used with --do-sample). Default: 0.95.")
parser.add_argument("--no-pin-memory", action="store_true",
                    help="Disable prefetch pin_memory (prefetch_pin_memory=False). "
                         "Eliminates the OS page-locking cost (~1.7s/layer at 70B scale) that "
                         "dominates total inference time (~89%% of wall time in profiling). "
                         "CPU→GPU DMA runs on pageable memory instead; CUDA handles the bounce "
                         "buffer, adding back only ~10–30ms/layer. Net effect: estimated 5–8× "
                         "total speedup for large models without any other changes.")
parser.add_argument("--profile", action="store_true",
                    help="Enable layer-streaming profiler (profiling_mode=True). "
                         "Prints per-step and aggregate breakdown of disk I/O, CPU->GPU transfer, "
                         "GPU compute, and decompression. Use to find the bottleneck. "
                         "Implies --max-new-tokens=3 unless explicitly set.")
args = parser.parse_args()

# When profiling, a short run is enough to characterise the bottleneck.
# Only apply the default cap when the user did not pass --max-new-tokens explicitly.
if args.profile and "--max-new-tokens" not in sys.argv:
    args.max_new_tokens = 3

compression = None if args.compression == "none" else args.compression

MAX_LENGTH = 128

# Use GPU if available, else CPU (e.g. in Docker without GPU or CI)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}  compression: {compression}  max_new_tokens: {args.max_new_tokens}")

# Auto-enable small-layer offload when using kv_cache_dir to avoid OOM on 8 GB with 72B
offload_small_layers = args.offload_small_layers or bool(args.kv_cache_dir)
t0 = time.perf_counter()
model = AutoModel.from_pretrained(
    args.model,
    device=device,
    compression=compression,
    cache_layers=args.cache_layers,
    kv_cache_dir=args.kv_cache_dir,
    use_gds=args.use_gds,
    offload_small_layers=offload_small_layers,
    offload_small_layers_use_cpu_cache=not args.no_offload_small_layers_cpu_cache,
    prefetch_pin_memory=not args.no_pin_memory,
    profiling_mode=args.profile,
)
load_s = time.perf_counter() - t0
extras = []
if args.kv_cache_dir:
    extras.append(f"kv_cache_dir={args.kv_cache_dir}")
if offload_small_layers:
    extras.append("offload_small_layers=True")
if args.use_gds:
    extras.append("use_gds=True")
if args.no_pin_memory:
    extras.append("prefetch_pin_memory=False")
extras_str = f"  [{', '.join(extras)}]" if extras else ""
print(f"[time] model load: {load_s:.2f}s{extras_str}")

system_content = "/no_think" if args.no_think else "You are a helpful assistant."
messages = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": args.prompt},
]

input_text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

input_tokens = model.tokenizer(
    [input_text],
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LENGTH,
)
input_ids = input_tokens["input_ids"].to(device)
attention_mask = input_tokens.get("attention_mask")
if attention_mask is None:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
else:
    attention_mask = attention_mask.to(device)

t1 = time.perf_counter()
generate_kwargs = dict(
    attention_mask=attention_mask,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
    do_sample=args.do_sample,
    return_dict_in_generate=True,
)
if args.do_sample:
    generate_kwargs["temperature"] = args.temperature
    generate_kwargs["top_p"] = args.top_p

generation_output = model.generate(input_ids, **generate_kwargs)
gen_s = time.perf_counter() - t1

input_len = input_tokens["input_ids"].shape[1]
num_tokens = generation_output.sequences.shape[1] - input_len
# Decode only the newly generated tokens, not the full conversation
output = model.tokenizer.decode(
    generation_output.sequences[0][input_len:], skip_special_tokens=True
)

print(output.strip())
print(f"[time] generate: {gen_s:.2f}s | new tokens: {num_tokens} | {num_tokens / gen_s:.1f} tok/s")

if args.profile:
    print()
    print("=" * 64)
    print("PROFILING SUMMARY  (aggregate across all forward passes)")
    print("=" * 64)
    model.print_profile_summary()
    print()
    print("How to read the table:")
    print("  Disk I/O         -> bound by SSD/HDD speed.  Fix: GDS, NVMe, cache-layers")
    print("  CPU load wait    -> prefetch not hiding disk latency.  Fix: tune prefetch")
    print("  CPU->GPU transfer-> PCIe bandwidth.  Fix: smaller dtype, compression")
    print("  Decompression    -> 4-bit/8-bit decode cost.  Fix: cache-layers, GDS+fp16")
    print("  GPU forward      -> compute bound.  Fix: flash-attention, larger batch")
