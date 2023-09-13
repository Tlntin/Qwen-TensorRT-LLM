import argparse
import os


now_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Benchmark the throughput.")
parser.add_argument(
    "--hf_model_dir",
    type=str,
    default=os.path.join(now_dir, "qwen_7b_chat")
)
parser.add_argument(
    "--tokenizer_dir",
    type=str,
    default=os.path.join(now_dir, "qwen_7b_chat")
)
parser.add_argument(
    '--ft_dir_path',
    type=str,
    default=os.path.join(now_dir, "c-model", "qwen_7b_chat", "1-gpu")
)
parser.add_argument(
    '--engine_dir',
    type=str,
    default=os.path.join(now_dir, "trt_engines", "fp16", "1-gpu")
)
parser.add_argument(
    "--hf_max_batch_size",
    type=int,
    default=1,
    help="Maximum batch size for HF backend."
)

parser.add_argument(
    "--trt_max_batch_size",
    type=int,
    default=2,
    help="Maximum batch size for TRT-LLM backend."
)
parser.add_argument(
    "--chat-format",
    type=str,
    default="chatml",
    choices=["chatml", "raw"],
    help="choice the model format, base or chat"
)
parser.add_argument(
    "--max_input_len",
    type=int,
    default=2048,
    help="Maximum input length."
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="Maximum number of generate new tokens."
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.5,
    help="Top p for sampling."
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1,
    help="Top k for sampling."
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for sampling."
)

args = parser.parse_args()
