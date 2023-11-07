from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no api"
)

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
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        top_p=0.5,
        temperature=0,
        n=1,
        max_tokens=4096,
        stream=False,
    )
    message = completion.choices[0].message
    response_text = message.content
    print('ChatBot: {}'.format(response_text))
    messages.append({"role": "assistant", "content": response_text})