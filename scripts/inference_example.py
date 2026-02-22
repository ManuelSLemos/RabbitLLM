import argparse
import time
import warnings

import torch
from rabbitllm import AutoModel

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
parser.add_argument("--no-think", action="store_true",
                    help="Disable Qwen3 chain-of-thought thinking mode (adds /no_think system prompt).")
parser.add_argument("--do-sample", action="store_true",
                    help="Use sampling instead of greedy decoding. Recommended for thinking models "
                         "to avoid repetition loops.")
parser.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (only used with --do-sample). Default: 0.6.")
parser.add_argument("--top-p", type=float, default=0.95,
                    help="Top-p nucleus sampling (only used with --do-sample). Default: 0.95.")
args = parser.parse_args()

compression = None if args.compression == "none" else args.compression

MAX_LENGTH = 128

# Use GPU if available, else CPU (e.g. in Docker without GPU or CI)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}  compression: {compression}  max_new_tokens: {args.max_new_tokens}")

t0 = time.perf_counter()
model = AutoModel.from_pretrained(
    args.model,
    device=device,
    compression=compression,
    cache_layers=args.cache_layers,
)
load_s = time.perf_counter() - t0
print(f"[time] model load: {load_s:.2f}s")

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
