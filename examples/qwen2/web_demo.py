import os
import gradio as gr
import mdtex2html
from default_config import default_config
from openai import OpenAI


client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no api"
)

now_dir = os.path.dirname(os.path.abspath(__file__))


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = [
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        ]
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


def predict(input_text, chatbot, top_p, temperature, max_generate_length, history):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for (message, response) in history:
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": input_text})
    chatbot.append((parse_text(input_text), ""))
    history.append((input_text, ""))
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        top_p=top_p,
        temperature=temperature,
        n=1,
        max_tokens=max_generate_length,
        stream=True,
    )
    response_text = ""
    for event in response:
        event_text = event.choices[0].delta.content  # extract the text
        if event_text is None:
            event_text = ""
        response_text += event_text
        chatbot[-1] = (parse_text(input_text), parse_text(response_text))
        history[-1] = (input_text, response_text)
        yield chatbot, history
    messages.append({"role": "assistant", "content": response_text})


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Qwen1.5-Chat (Power By TensorRT-LLM)</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(
                    show_label=False,
                    placeholder="Input...",
                    lines=10,
                    container=False
                )
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            top_p = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.8,
                step=0.1,
                label="top-p",
                interactive=True
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                value=1,
                step=0.1,
                label="temperature",
                interactive=True
            )
            max_generate_length = gr.Slider(
                0,
                default_config.max_new_tokens,
                value=default_config.max_new_tokens // 2,
                step=1.0,
                label="Maximum generate length", interactive=True
            )

    history = gr.State([])

    submitBtn.click(
        predict,  # call function
        [user_input, chatbot, top_p, temperature, max_generate_length, history], # inputs
        [chatbot, history],  # outputs
        show_progress=True,
    )
    # reset input
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", share=True, inbrowser=False)
# demo.queue().launch(server_name="localhost", share=False, inbrowser=False)
