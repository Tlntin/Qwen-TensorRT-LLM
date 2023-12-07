import asyncio
import json
import aiohttp_sse_client.client
from aiohttp import ClientSession
from aiohttp_sse_client import client as sseclient


async def handle_event(event: aiohttp_sse_client.client.MessageEvent, event_source):
    # 处理 SSE 事件的回调函数
    data = json.loads(event.data)
    # print("data", data)
    if event.type == "finish":
        try:
            await event_source.close()
        except Exception as err:
            print("close with error", err)
    return data["response"], event.type


async def listen_sse(query, history=None, max_new_tokens=4096, top_p=0.5, temperature=0):
    if history is None:
        history = []
    async with ClientSession() as session:
        url = 'http://127.0.0.1:8000/stream_chat/'
        data = {
            "query": query,
            "history": history,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
        }
        headers = {'Content-Type': 'application/json'}
        response = ""
        if history is None:
            history = []
        print("Chatbox: ", end='', flush=True)
        async with sseclient.EventSource(url, json=data, headers=headers, session=session) as event_source:
            try:
                async for event in event_source:
                    # 将事件传递给回调函数进行处理
                    new_text, e_type = await handle_event(event, event_source)
                    print(new_text, end='', flush=True)
                    response += new_text
                    if e_type == "finish":
                        break
            except Exception as err:
                print("event close", err)
        print("")
        history.append((query, response))
        return response, history


if __name__ == "__main__":
    history1 = []
    print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
    while True:
        query = input("Human: ")
        if query == 'exit':
            break
        if query == 'clear':
            history1 = []
            continue
        _, history1 = asyncio.run(listen_sse(query, history1))
