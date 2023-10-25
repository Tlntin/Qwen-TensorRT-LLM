### Triton部署TensorRT-LLM

1. 部署Qwen-7B-Chat-TensorRT-LLM , 参考该项目：https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM ，需要部署api。


2. 下载Langchain-Chatchat，当前最新版0.2.6
```bash
git clone https://github.com/chatchat-space/Langchain-Chatchat -b v0.2.6
```
- 环境配置安装readme操作即可。
- 模型下载可以忽略，如果网络好的话，可以在线下载。
- 初始化配置，参考readme操作即可。
```bash
python copy_config_example.py
```

3. 修改模型配置文件`configs/model_config.py`，修改`LLM_MODEL`为`OpenAI`
- 修改前
```bash
# LLM 名称
LLM_MODEL = "chatglm2-6b"
```

- 修改后
```bash
# LLM 名称
LLM_MODEL = "OpenAI"
```

4. 修改模型配置文件`configs/model_config.py`，修改OpenAI的url地址为你部署TensorRT-LLM api的地址
- 修改前
```bash
"OpenAI": {
        "model_name": "your openai model name(such as gpt-4)",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "your OPENAI_API_KEY",
        "openai_proxy": "",
    },
```

- 修改后
```bash
"OpenAI": {
        "model_name": "gpt-3.5-turbo",
        "api_base_url": "http://127.0.0.1:8000/v1",
        "api_key": "",
        "openai_proxy": "",
    },
```

5. 初始化启动数据
```bash
python init_database.py --recreate-vs
```

6. 启动Langchain-Chatchat，会自动打开浏览器
```bash
python startup.py -a
```

7. 再选择LLM模型部分，选择`OpenAI (Running)`即可，然后就可以愉快的聊天了。

8. 如果要知识库问答。
- 先选择`知识库管理`，新建知识库，然后上传任意一个文档上去，推荐点击一下`根据源文件重建向量库`。
- 回到对话，对话模式选择`知识库问答`，LLM模型选择`OpenAI(Running)`，最下面的知识库，选择你刚刚新建的那个，然后即可在右边愉快的问答了。
