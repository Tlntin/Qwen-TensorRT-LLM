import openai
openai.api_base = "http://localhost:8000/v1"
openai.api_key = ""

 
messages = [{"role": "system", "content": "You are a helpful assistant."}]
print("欢迎使用Qwen聊天机器人，输入exit退出，输入clear清空历史记录")
while True:
    prompt = input('Human：')
    if prompt == 'exit':
        break
    if prompt == 'clear':
        messages = messages[:1]
        continue
    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        top_p=0.5,
        temperature=0,
        n=1,
        stream=False,
    )
    message = completion.choices[0].message
    print('ChatBot：', message["content"])