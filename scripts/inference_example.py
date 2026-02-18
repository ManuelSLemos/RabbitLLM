import time
import warnings

import torch
from rabbitllm import AutoModel

MAX_LENGTH = 128

# Use GPU if available, else CPU (e.g. in Docker without GPU or CI)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*CUDA.*unknown error.*", category=UserWarning)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# eager is often faster on small models/some GPUs; try "auto" or "sdpa" to compare
t0 = time.perf_counter()
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device=device)
load_s = time.perf_counter() - t0
print(f"[time] model load: {load_s:.2f}s")

# Qwen2.5-Instruct models require the ChatML template format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
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
generation_output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    use_cache=True,
    do_sample=False,
    return_dict_in_generate=True,
)
gen_s = time.perf_counter() - t1

input_len = input_tokens["input_ids"].shape[1]
num_tokens = generation_output.sequences.shape[1] - input_len
# Decode only the newly generated tokens, not the full conversation
output = model.tokenizer.decode(
    generation_output.sequences[0][input_len:], skip_special_tokens=True
)

print(output.strip())
print(f"[time] generate: {gen_s:.2f}s | new tokens: {num_tokens} | {num_tokens / gen_s:.1f} tok/s")
