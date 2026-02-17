import time
from rabbitllm import AutoModel

MAX_LENGTH = 128

# eager is often faster on small models/some GPUs; try "auto" or "sdpa" to compare
t0 = time.perf_counter()
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", attn_implementation="flash_attention_2")
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
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH)

t1 = time.perf_counter()
generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=50,
    use_cache=True,
    do_sample=False,
    return_dict_in_generate=True)
gen_s = time.perf_counter() - t1

num_tokens = generation_output.sequences.shape[1] - input_tokens['input_ids'].shape[1]
output = model.tokenizer.decode(
    generation_output.sequences[0], skip_special_tokens=True)

print(output)
print(f"[time] generate: {gen_s:.2f}s | new tokens: {num_tokens} | {num_tokens/gen_s:.1f} tok/s")
