import asyncio
import datetime
import json
import time
from typing import List, Literal, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from run import get_model, QWenForCausalLMGenerationSession
from cli_chat import parse_arguments
from args import args as raw_args


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    runtime_mapping,
)


@app.get("/")
async def root():
    return "Hello! This is QWen-Chat-7B API."


class Data(BaseModel):
    query: str
    system: str = "You are a helpful assistant."
    history: List[List[str]] = [],
    max_input_length: Optional[int] = raw_args.max_input_len
    max_new_tokens: Optional[int] = raw_args.max_new_tokens
    top_p: Optional[float] = raw_args.top_p
    temperature: Optional[float] = raw_args.temperature


@app.post("/chat/")
async def create_item(data: Data):
    if not isinstance(data.query, str) or len(data.query) == 0:
        return HTTPException(status_code=400, detail="Invalid request")
    # if you want to change this, you need to change the max_input_len/max_output_len in tensorrt_llm_july-release-v1/examples/qwen/build.py
    max_input_length = min(data.max_input_length, raw_args.max_input_len)
    max_new_tokens = min(data.max_new_tokens, raw_args.max_new_tokens)
    sampling_config.top_p = data.top_p 
    sampling_config.temperature = data.temperature
    history = data.history
    response = decoder.chat(
        tokenizer=tokenizer,
        sampling_config=sampling_config,
        input_text=data.query,
        system_text=data.system,
        history=history,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens
    )
    history += [(data.query, response[0])]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response[0],
        "history": history,
        "status": 200,
        "time": time,
    }
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    # print(log)
    return answer


@app.get("/stream_chat/")
async def stream_chat(request: Request):
    global tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get("query")
    if not isinstance(query, str) or len(query) == 0:
        return HTTPException(status_code=400, detail="Invalid request")
    system = json_post_list.get("system", "You are a helpful assistant.")
    history = json_post_list.get("history", [])
    max_input_length = json_post_list.get("max_input_length", raw_args.max_input_len)
    max_new_tokens = json_post_list.get("max_new_tokens", raw_args.max_new_tokens)
    sampling_config.top_p = json_post_list.get("top_p", raw_args.top_p)
    sampling_config.temperature = json_post_list.get("temperature", raw_args.temperature)
    # if you want to change this, you need to change the max_input_len/max_output_len in tensorrt_llm_july-release-v1/examples/qwen/build.py
    max_input_length = min(max_input_length, raw_args.max_input_len)
    max_new_tokens = min(max_new_tokens, raw_args.max_new_tokens)
    STREAM_DELAY = 1  # second
    RETRY_TIMEOUT = 15000  # milisecond

    async def event_generator(
            query, system, history, max_input_length,
            max_output_length, sampling_config
        ):
        for response in decoder.chat_stream(
            tokenizer=tokenizer,
            sampling_config=sampling_config,
            input_text=query,
            system_text=system,
            history=history,
            max_input_length=max_input_length,
            max_new_tokens=max_output_length
        ):
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            # Checks for new messages and return them to client if any
            try:
                temp_dict = {
                    "response": response[0],
                    "history": history + [(query, response[0])],
                    "finish": False,
                }
                yield {
                    "event": "new_message",
                    "id": "message_id",
                    "retry": RETRY_TIMEOUT,
                    "data": json.dumps(temp_dict, ensure_ascii=False),
                }
            except StopIteration:
                await asyncio.sleep(STREAM_DELAY)
        temp_dict = {
            "response": response[0],
            "history": history + [(query, response[0])],
            "finish": True,
        }
        yield {
            "event": "finish",
            "id": "finish_id",
            "retry": RETRY_TIMEOUT,
            "data": json.dumps(temp_dict, ensure_ascii=False),
        }

    return EventSourceResponse(
        event_generator(
            query, system, history, max_input_length,
            max_new_tokens, sampling_config
        )
    )


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
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = raw_args.temperature
    top_p: Optional[float] = raw_args.top_p
    max_tokens: Optional[int] = raw_args.max_new_tokens
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


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


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    sampling_config.top_p = request.top_p
    sampling_config.temperature = request.temperature
    max_new_tokens = min(request.max_tokens, raw_args.max_new_tokens)
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system = prev_messages.pop(0).content
    else:
        system = "You are a helpful assistant."

    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if (
                prev_messages[i].role == "user"
                and prev_messages[i + 1].role == "assistant"
            ):
                history.append(
                    [
                        prev_messages[i].content,
                        prev_messages[i + 1].content,
                    ]
                )

    if request.stream:
        generate = predict(query, system, history, max_new_tokens, request.model)
        return EventSourceResponse(
            generate, media_type="text/event-stream"
        )

    query_text = query.lstrip("\n").strip()
    response = decoder.chat(
        tokenizer=tokenizer,
        sampling_config=sampling_config,
        input_text=query_text,
        system_text=system,
        history=history,
        max_new_tokens=max_new_tokens,
    )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response[0]),
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
    )


async def predict(query: str, system: str, history: List[List[str]], max_new_tokens,  model_id: str):

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
        chunk.json(exclude_unset=True, ensure_ascii=False)
    )

    old_position = 0
    for response in decoder.chat_stream(
        tokenizer=tokenizer,
        sampling_config=sampling_config,
        input_text=query,
        system_text=system,
        history=history,
        max_new_tokens=max_new_tokens,
    ):
        if len(response) == 0:
            continue
        new_text = response[0][old_position:]
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
            chunk.json(exclude_unset=True, ensure_ascii=False)
        )
        old_position = len(response[0])

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(
        chunk.json(exclude_unset=True, ensure_ascii=False)
    )
    yield "[DONE]"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

