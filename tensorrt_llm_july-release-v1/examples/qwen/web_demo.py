import torch
import os
import gradio as gr
import mdtex2html
import os
import torch
import numpy as np
from run import QWenForCausalLMGenerationSession
from run import get_model
from cli_chat import parse_arguments


now_dir = os.path.dirname(os.path.abspath(__file__))

args = parse_arguments()
(
    model_config, sampling_config, runtime_mapping, runtime_rank,
    serialize_path, remove_input_padding, 
    tokenizer, eos_token_id, pad_token_id
) = get_model(args.tokenizer_dir, args.engine_dir, args.log_level)

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


def predict(input_text, chatbot, max_input_length, max_generate_length, history):
    chatbot.append((parse_text(input_text), ""))
    history.append((input_text, ""))
    for response in decoder.chat_stream(
        tokenizer=tokenizer,
        sampling_config=sampling_config,
        input_text=input_text,
        history=history,
        max_input_len=max_input_length,
        max_output_len=max_generate_length,
    ):
        if response is None:
            break
        chatbot[-1] = (parse_text(input_text), parse_text(response))
        history[-1] = (input_text, response)
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
            max_input_length = gr.Slider(0, 2048, value=2048, step=1.0, label="Maximum input length", interactive=True)
            max_generate_length = gr.Slider(0, 512, value=512, step=1.0, label="Maximum generate length", interactive=True)

    history = gr.State([])

    submitBtn.click(predict, [user_input, chatbot, max_input_length, max_generate_length, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
