from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from default_config import default_config
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument(
    "--hf_model_dir",
    type=str,
    default=default_config.hf_model_dir,
)
parser.add_argument(
    '--tokenizer_dir',
    type=str,
    default=default_config.tokenizer_dir,
    help="Directory containing the tokenizer.model."
)
parser.add_argument(
    "--quant_ckpt_path",
    type=str,
    default=os.path.join(
        default_config.int4_gptq_model_dir,
    ),
)


args = parser.parse_args()
# model_id_or_path = default_config.hf_model_dir
# quantized_model_dir = default_config.int4_gptq_model_dir
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_dir, use_fast=True, trust_remote_code=True
)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    ),
]
quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    true_sequential=True,
)

print("model_path", args.hf_model_dir)
model = (
    AutoGPTQForCausalLM.from_pretrained(
        args.hf_model_dir, quantize_config, trust_remote_code=True, use_flash_attn=False
    )
    .eval()
    # .cuda()
)
print("loading model to run gptq, may need few minute...")
# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)
print("quantized ok!")

# save quantized model
model.save_quantized(args.quant_ckpt_path, use_safetensors=True)