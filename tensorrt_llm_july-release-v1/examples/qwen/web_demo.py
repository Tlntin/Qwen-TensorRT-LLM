import torch
import os
import sys
import gradio as gr
import mdtex2html
from threading import Thread
from transformers import AutoTokenizer
import os
import json
import torch
import tensorrt_llm
import argparse
import numpy as np
from transformers import AutoTokenizer

import tensorrt_llm
from typing import List
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import _tile_beam_width
from build import get_engine_name  # isort:skip
from run import QWenForCausalLMGenerationSession
from utils.utils import make_context


now_dir = os.path.dirname(os.path.abspath(__file__))



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=os.path.join(now_dir, 'trt_engines', 'fp16', '1-gpu')
    )
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        default=os.path.join(now_dir, 'qwen_7b_chat'),
        help="Directory containing the tokenizer.model."
    )
    parser.add_argument(
        '--stream',
        type=bool,
        default=True,
        help="return text with stream")
    return parser.parse_args()


args = parse_arguments()


# --load the tokenizer and engine #
tensorrt_llm.logger.set_level(args.log_level)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_dir,
    legacy=False,
    trust_remote_code=True,
)
config_path = os.path.join(args.engine_dir, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
gen_config_path = os.path.join(args.tokenizer_dir, 'generation_config.json')
with open(gen_config_path, 'r') as f:
    gen_config = json.load(f)
top_k = gen_config['top_k']
top_p = gen_config['top_p']
chat_format = gen_config['chat_format']
if chat_format == "raw":
    eos_token_id = gen_config['eos_token_id']
    pad_token_id = gen_config['pad_token_id']
elif chat_format == "chatml":
    pad_token_id = eos_token_id = tokenizer.im_end_id
else:
    raise Exception("unkown chat format ", chat_format)

use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
remove_input_padding = config['plugin_config']['remove_input_padding']
dtype = config['builder_config']['precision']
world_size = config['builder_config']['tensor_parallel']
assert world_size == tensorrt_llm.mpi_world_size(), \
    f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
num_heads = config['builder_config']['num_heads'] // world_size
hidden_size = config['builder_config']['hidden_size'] // world_size
vocab_size = config['builder_config']['vocab_size']
num_layers = config['builder_config']['num_layers']
multi_query_mode = config['builder_config']['multi_query_mode']

runtime_rank = tensorrt_llm.mpi_rank()
runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

model_config = ModelConfig(
    num_heads=num_heads,
    hidden_size=hidden_size,
    vocab_size=vocab_size,
    num_layers=num_layers,
    gpt_attention_plugin=use_gpt_attention_plugin,
    multi_query_mode=multi_query_mode,
    remove_input_padding=remove_input_padding
)
sampling_config = SamplingConfig(
    end_id=eos_token_id,
    pad_id=pad_token_id,
    num_beams=1,
    top_k = top_k,
    top_p = top_p,
)

engine_name = get_engine_name('qwen', dtype, world_size, runtime_rank)
serialize_path = os.path.join(args.engine_dir, engine_name)
print(f'Loading engine from {serialize_path}')
with open(serialize_path, 'rb') as f:
    engine_buffer = f.read()
decoder = QWenForCausalLMGenerationSession(
    model_config,
    engine_buffer,
    runtime_mapping
)



"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def generate_stream(query,
                    history,
                    max_input_length,
                    max_output_len):
    if history is None:
        history = []
    _, input_id_list = make_context(
        tokenizer=tokenizer,
        query=query,
        history=history,
    )
    input_ids = torch.from_numpy(
        np.array(input_id_list, dtype=np.int32)
    ).type(torch.int32).unsqueeze(0).cuda()
    input_lengths = torch.cuda.IntTensor([input_ids.size(1)])
    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0), max_input_length, max_output_len)
    history.append((query, ""))
    for (output_ids, end_step) in decoder.steam_decode(input_ids, input_lengths, sampling_config):
        torch.cuda.synchronize()
        if runtime_rank == 0:
            output_begin = max_input_length
            output_end = max_input_length + end_step
            outputs = output_ids[0][0][output_begin: output_end].tolist()
            output_text = tokenizer.decode(outputs, skip_special_tokens=True)
            history[-1] = (query, output_text)
            yield output_text, history


def predict(input, chatbot, max_input_length, max_generate_length, history):
    chatbot.append((parse_text(input), ""))
    for response, history in generate_stream(
            input,
            history,
            max_input_length=max_input_length,
            max_output_len=max_generate_length,
    ):
        if response is None:
            break
        chatbot[-1] = (parse_text(input), parse_text(response))

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Qwen-7B-Chat (Power By TensorRT-LLM)</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_input_length = gr.Slider(0, 1024, value=512, step=1.0, label="Maximum input length", interactive=True)
            max_generate_length = gr.Slider(0, 2048, value=512, step=1.0, label="Maximum generate length", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_input_length, max_generate_length, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
