import asyncio
import datetime
import os.path
import torch
import json
import time
from typing import List, Literal, Optional, Union, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from run import Vit, QWenInfer, parse_arguments
from vit_onnx_trt import Preprocss
from default_config import default_config
import copy
import re


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
args = parse_arguments()
image_preprocess = Preprocss(image_size=448)
vit = Vit(args.vit_engine_dir, args.log_level)
qinfer = QWenInfer(args.tokenizer_dir,args.qwen_engine_dir, args.log_level)
qinfer.qwen_model_init()


@app.get("/")
async def root():
    return "Hello! This is QWen-Chat-7B API."


# --- Compatible with OpenAI ChatGPT --- #
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: Optional[str]
    images: Optional[List[str]] = []
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = ""
    images: Optional[List[str]] = []


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = default_config.temperature
    top_p: Optional[float] = default_config.top_p
    max_tokens: Optional[int] = default_config.max_new_tokens
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[
            ChatCompletionResponseChoice,
            ChatCompletionResponseStreamChoice,
        ]
    ]
    created: Optional[int] = Field(
        default_factory=lambda: int(time.time())
    )


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip("\n")
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


def parse_messages(messages, functions):
    if all(m.role != "user" for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = copy.deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == "system":
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }

    _messages = messages
    messages = []
    image_list = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if len(m.images) > 0:
            image_list.extend(m.images)
        if content:
            content = content.lstrip("\n").rstrip()
        if role == "function":
            if (len(messages) == 0) or (messages[-1].role != "assistant"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1].content += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
        elif role == "assistant":
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0
            if func_call is None:
                if functions:
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                f_name, f_args = func_call["name"], func_call["arguments"]
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."
                content = f"\n{content}\nAction: {f_name}\nAction Input: {f_args}"
            if messages[-1].role == "user":
                messages.append(
                    ChatMessage(role="assistant", content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1].content += content
        elif role == "user":
            messages.append(
                ChatMessage(role="user", content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == "user":
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == "user" and messages[i + 1].role == "assistant":
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t) :]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
    return query, history, image_list


def parse_response(response):
    func_name, func_args = "", ""
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        k = response.rfind("\nObservation:")
        func_name = response[i + len("\nAction:") : j].strip()
        func_args = response[j + len("\nAction Input:") : k].strip()
    if func_name:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role="assistant",
                content=response[:i],
                function_call={"name": func_name, "arguments": func_args},
            ),
            finish_reason="function_call",
        )
        return choice_data
    z = response.rfind("\nFinal Answer: ")
    if z >= 0:
        response = response[z + len("\nFinal Answer: ") :]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )
    return choice_data


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, sampling_config):
    im_end = qinfer.tokenizer.im_end
    output = qinfer.qwen_infer(
        None,
        None,
        input_text="",
        history=history,
    )
    output = trim_stop_words(output, ["<|endoftext|>", im_end])
    # print(f"<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>")
    return output


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    # print("Debug, top_p: ", request.top_p)
    # print("Debug, temperature: ", request.temperature)
    # print("Debug, max_tokens: ", request.max_tokens)
    # if request.top_p is not None:
    #     sampling_config.top_p = request.top_p
    # else:
    #     sampling_config.top_p = default_config.top_p
    # if request.temperature is not None:
    #     sampling_config.temperature = request.temperature
    # else:
    #     sampling_config.temperature = default_config.temperature
    if request.max_tokens is not None:
        max_new_tokens = min(request.max_tokens, default_config.max_new_tokens)
    else:
        max_new_tokens = default_config.max_new_tokens
    if request.messages[-1].role not in ["user", "function"]:
        print(454)
        raise HTTPException(status_code=400, detail="Invalid request")
    # query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content
    else:
        system = "You are a helpful assistant."

    # history = []
    # if len(prev_messages) % 2 == 0:
    #     for i in range(0, len(prev_messages), 2):
    #         if (
    #             prev_messages[i].role == "user"
    #             and prev_messages[i + 1].role == "assistant"
    #         ):
    #             history.append(
    #                 [
    #                     prev_messages[i].content,
    #                     prev_messages[i + 1].content,
    #                 ]
    #             )
    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if "Observation:" not in stop_words:
            stop_words.append("Observation:")

    query, history, images = parse_messages(request.messages, request.functions)
    # print("Debug query: ", query)
    # print("Debug history: ", history)
    # print("Debug images: ", images)

    # Todo process image url
    image_list = [
        image for image in images
        if os.path.exists(image)
    ]
    if len(images) > 0:
        images = image_preprocess.encode(image_list)
        images_path = [{"image": image} for image in image_list]
        input_vit = vit.run(images=images)
    else:
        images_path = None
        input_vit = None

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Function calling is not yet implemented for stream mode.",
            )
        generate = predict(
            query,
            system,
            history,
            max_new_tokens,
            request.model,
            input_vit=input_vit,
            images_path=images_path,
        )
        return EventSourceResponse(
            generate, media_type="text/event-stream"
        )
    # stop_words_ids = [tokenizer.encode(s) for s in stop_words] if stop_words else None
    # print("gen kwargs",gen_kwargs)
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(
            history,
            # stop_words_ids=stop_words_ids,
            # sampling_config=sampling_config,
        )
    else:
        query_text = query.lstrip("\n").strip()
        response = qinfer.qwen_infer(
            input_text=query_text,
            max_new_tokens=max_new_tokens,
            history=history,
            input_vit=input_vit,
            images_path=images_path,
        )
    response = trim_stop_words(response, stop_words)
    if request.functions:
        choice_data = parse_response(response)
    else:
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason="stop",
        )

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
    )


async def predict(
    query: str,
    system: str,
    history: List[List[str]],
    max_new_tokens,
    model_id: str,
    input_vit=None,
    images_path=None
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None,
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(
        chunk.model_dump_json(exclude_unset=True)
    )
    # if images_path is not None:
    #     content_list = images_path + [{'text': query}]
    #     query = qinfer.tokenizer.from_list_format(content_list)
    # print("Debug system", system)
    # print("Debug query", query)
    # print("Debug history", history)
    for new_text in qinfer.qwen_infer_stream(
        input_text=query,
        history=history,
        system=system,
        max_new_tokens=max_new_tokens,
        input_vit=input_vit,
    ):
        if len(new_text) == 0:
            continue
        # print("Debug, new_text: ", new_text)
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_text),
            finish_reason=None,
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk",
        )
        yield "{}".format(
            chunk.model_dump_json(exclude_unset=True)
        )

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(
        chunk.model_dump_json(exclude_unset=True)
    )
    yield "[DONE]"


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
    uvicorn.run(app, host="localhost", port=8000, workers=1)

