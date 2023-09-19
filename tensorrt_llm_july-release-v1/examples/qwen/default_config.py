import os


class DefaultConfig:
    now_dir = os.path.dirname(os.path.abspath(__file__))
    hf_model_dir = os.path.join(now_dir, "qwen_7b_chat")
    tokenizer_dir = os.path.join(now_dir, "qwen_7b_chat")
    ft_dir_path = os.path.join(now_dir, "c-model", "qwen_7b_chat")
    engine_dir=os.path.join(now_dir, "trt_engines", "fp16", "1-gpu")

    # Maximum batch size for HF backend.
    hf_max_batch_size = 1

    # Maximum batch size for TRT-LLM backend.
    trt_max_batch_size=1

    # choice the model format, base or chat
    #  choices=["chatml", "raw"],
    chat_format = "chatml"

    # Maximum input length.
    max_input_len = 2048

    # Maximum number of generate new tokens.
    max_new_tokens = 2048


    # Top p for sampling.
    top_p = 0.5


    # Top k for sampling.
    top_k = 1

    # Temperature for sampling.
    temperature = 1.0


default_config = DefaultConfig()