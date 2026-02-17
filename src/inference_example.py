from airllm import AutoModel

MAX_LENGTH = 128

model = AutoModel.from_pretrained("Qwen/Qwen-7B")

input_text = ['What is the capital of China?',]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=5,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
