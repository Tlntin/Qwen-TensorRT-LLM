import json
import requests


def chat(query, history=None, max_length=512, top_p=0.5, temperature=0):
    if history is None:
        history = []
    url = 'http://127.0.0.1:8000/chat/'
    data = {
        "query": query,
        "history": history,
        "max_length": max_length,
        "top_p": top_p,
        "temperature": temperature,
    }
    headers = {'Content-Type': 'application/json'}
    res = requests.post(url=url, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        data = res.json()
        if data["status"] == 200:
            return data["response"], data["history"]
        else:
            print("Error: ", data)
            return "", history
    else:
        print("Error: ", res.status_code)
        return "", history
     


if __name__ == "__main__":
    history1 = []
    print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
    while True:
        query = input("Human:")
        if query == 'exit':
            break
        if query == 'clear':
            history1 = []
            continue
        response, history1 = chat(query, history1)
        print("ChatBot:" + response)
        