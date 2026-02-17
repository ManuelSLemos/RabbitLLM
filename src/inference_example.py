from rabbitllm import AutoModel

MAX_LENGTH = 128

model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", attn_implementation="eager")

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

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=50,
    use_cache=True,
    do_sample=False,
    return_dict_in_generate=True)

output = model.tokenizer.decode(
    generation_output.sequences[0], skip_special_tokens=True)
print(output)
