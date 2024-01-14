import os


class DefaultConfig:
    now_dir = os.path.dirname(os.path.abspath(__file__))
    hf_model_dir = os.path.join(now_dir, "Qwen-VL-Chat")
    tokenizer_dir = os.path.join(now_dir, "Qwen-VL-Chat")
    int4_gptq_model_dir = os.path.join(now_dir, "qwen_7b_vl_chat_int4")
    ft_dir_path = os.path.join(now_dir, "c-model", "Qwen-VL-Chat")
    qwen_engine_dir=os.path.join(now_dir, "trt_engines", "Qwen-VL-7B-fp16")
    vit_engine_dir=os.path.join(now_dir, "plan")

    # Maximum batch size for HF backend.
    hf_max_batch_size = 1

    # Maximum batch size for TRT-LLM backend.
    trt_max_batch_size = 4

    # choice the model format, base or chat
    #  choices=["chatml", "raw"],
    chat_format = "chatml"

    # Maximum input length.
    max_input_len = 1024 * 2

    # Maximum number of generate new tokens.
    max_new_tokens = 512

    # Top p for sampling.
    top_p = 0.8


    # Top k for sampling.
    top_k = 0

    # Temperature for sampling.
    temperature = 1.0


default_config = DefaultConfig()
