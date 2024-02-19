# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer
from default_config import default_config
device = "cuda"     # the device to load the model onto


model = Qwen2ForCausalLM.from_pretrained(
    # "Qwen/Qwen1.5-72B-Chat",
    default_config.hf_model_dir,
    device_map="auto"
).half()
tokenizer = Qwen2Tokenizer.from_pretrained(default_config.hf_model_dir)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请问你叫什么？"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("Input Text: ", text)
input_ids = tokenizer([text], return_tensors="pt").to(device).input_ids
print("Input Shape: ", input_ids.shape)

generated_ids = model.generate(
    input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Response: ", response)