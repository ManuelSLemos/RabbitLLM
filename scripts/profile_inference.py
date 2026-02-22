#!/usr/bin/env python3
"""
Profile layer-streaming inference to find the bottleneck (disk, CPU→GPU, or forward).

Use with large models (e.g. 70B) to see where time is spent. After generate(), the
profiler prints total time per category. See docs/TROUBLESHOOTING.md for how to
interpret the metrics and what to optimize.

Usage:
    uv run python scripts/profile_inference.py --model /path/to/70B-or-repo --max-new-tokens 20
    uv run python scripts/profile_inference.py --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 10
"""

import argparse
import logging
import time
import warnings

import torch
from rabbitllm import AutoModel

# Ensure profiler output is visible
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Profile inference to find bottleneck (disk / CPU→GPU / forward)")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID (e.g. .../70B or Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max new tokens to generate (default 20)")
    parser.add_argument("--device", default=None, help="Device (default: cuda:0 if available else cpu)")
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"],
                        help="Attention implementation (default auto)")
    parser.add_argument("--no-prefetch-pin-memory", action="store_true",
                        help="Disable pin_memory in prefetch (recommended for 70B/72B to reduce total time)")
    parser.add_argument("--compression", default=None, choices=["4bit", "8bit"],
                        help="Load 4-bit or 8-bit quantized split (requires prior split_and_save_layers with same compression)")
    parser.add_argument("--cache-layers", type=int, default=None,
                        help="Keep N layer state-dicts in CPU RAM between forward passes. "
                             "Cache hits skip disk I/O; pin_memory runs on already-in-RAM tensors "
                             "(RAM→pinned copy ~0.017 s/layer vs disk+pin ~0.67 s/layer). "
                             "Suggested: 30 for 4-bit 72B on 32 GB RAM.")
    parser.add_argument("--token", default=None, help="HuggingFace token for gated repos")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model with profiling_mode=True (device=%s)...", device)
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        device=device,
        profiling_mode=True,
        attn_implementation=args.attn_implementation,
        prefetch_pin_memory=not args.no_prefetch_pin_memory,
        compression=args.compression,
        cache_layers=args.cache_layers,
        token=args.token,
    )
    load_sec = time.perf_counter() - load_start
    logger.info("Model loaded in %.2fs", load_sec)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer in one short sentence."},
    ]
    if hasattr(model.tokenizer, "apply_chat_template"):
        input_text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = "What is 2+2? Answer in one short sentence."

    inputs = model.tokenizer(
        [input_text],
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

    logger.info("Running generate(max_new_tokens=%s, use_cache=True)...", args.max_new_tokens)
    gen_start = time.perf_counter()
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        do_sample=False,
        return_dict_in_generate=True,
    )
    gen_sec = time.perf_counter() - gen_start
    n_new = out.sequences.shape[1] - input_ids.shape[1]
    tok_s = n_new / gen_sec if gen_sec > 0 else 0
    logger.info("Generate finished: %.2fs, %s new tokens, %.2f tok/s", gen_sec, n_new, tok_s)

    # Profiler already printed per-category times; summarize
    logger.info("Check logs above for: load_safe_tensor, create_layer_from_state_dict, forward_per_layer, load_safe_tensor_cpu_wait")
    logger.info("See docs/TROUBLESHOOTING.md 'Finding the bottleneck with the profiler' for how to interpret and optimize.")


if __name__ == "__main__":
    main()
