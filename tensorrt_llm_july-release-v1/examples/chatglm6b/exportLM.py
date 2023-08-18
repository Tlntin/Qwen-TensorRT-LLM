import transformers

prompt = "续写：RTX4090具有760亿个晶体管，16384个CUDA核心"

# run the original model to export LM
tokenizer = transformers.AutoTokenizer.from_pretrained("pyTorchModel",
                                                       trust_remote_code=True)
model = transformers.AutoModel.from_pretrained(
    "pyTorchModel", trust_remote_code=True).half().cuda()

response, history = model.chat(tokenizer, prompt, history=[])
#print(response)
