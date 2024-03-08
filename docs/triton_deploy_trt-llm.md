# Triton23.12部署TensorRT-LLM,实现http查询

### 选择正确的环境
1. 选择版本。查询nvidia[官方文档](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)，可以看到目前最新的容器是23.12。
- 在**NVIDIA Driver**这一行，它推荐的英伟达驱动版本是545以上，对于数据卡，可以适当降低。如果你是游戏卡，驱动版本没有545，也不想升级，那么建议至少不要低太多，比如535其实也可以。
![020016f0975ebec6eae195bf77b61044.png](https://s2.loli.net/2024/03/08/anRjJh4AeIF6icf.png)
- 在**Triton Inference Server**这一行，可以看到它内置了triton server版本是2.41，需要的TensorRT-LLM版本是0.7.0。
![7ea2f2f9645f9019920a6704a91cd7c3.png](https://s2.loli.net/2024/03/08/Ou47LTKDyXidmS2.png)
2. 拉取镜像。进入[Nvidia镜像中心](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)找到tritonserver的镜像，选择和TensorRT-LLM（简称trtllm）有关的容器，然后拷贝镜像地址，最后使用`docker pull`来拉取该镜像。
![8adef9e1313d515b8144b2813f20d582.png](https://s2.loli.net/2024/03/08/2x6Swubef9QAdCV.png)
```bash
docker pull nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3
```
- 测试发现这个容器部署的时候会有问题，自己编译官方容器反而就可以，原因貌似是tritonserver目前只能用2.39而不能用2.41，参考[issues/246](https://github.com/triton-inference-server/tensorrtllm_backend/issues/246)，下面是编译命令。
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.7.0
git lfs install
git submodule update --init --recursive

# Use the Dockerfile to build the backend in a container
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .
# For aarch64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm --build-arg TORCH_INSTALL_TYPE="src_non_cxx11_abi" -f dockerfile/Dockerfile.trt_llm_backend .

# tag
docker tag triton_trt_llm triton_trt_llm:v0.7.0

# 回到上一层目录
cd ..
```
3. 拉取TensorRT-LLM的项目。
- 可以选择官方项目，但是注意要是v0.7.0
```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git -b v0.7.0
```
- 也可以选择我的项目，目前main分支就是0.7.0，后续可能会打成tag，建议实际访问项目地址，查看是否有0.7.0的tag。
```bash
git clone https://github.com/Tlntin/Qwen-TensorRT-LLM -b v0.7.0
```
- 下面演示是以我的项目为主，在triton_server上面部署Qwen-1.8B-Chat（毕竟这个模型比较小）
4. 拉取tensorrtllm_backend。这个是用来编排tensorrt-llm服务的，需要和TensorRT-LLM版本一致，这里同样选择0.7.0（第二步如果是手动编译的容器，就可以省略该步骤）
```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.7.0
```
5. 启动tritonserver容器
- 如果用官方镜像（目前有bug，部署不了）
```bash
docker run -d \
    --name triton \
    --net host \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v ${PWD}/tensorrtllm_backend:/tensorrtllm_backend \
    -v ${PWD}/Qwen-TensorRT-LLM/examples/qwen:/root/qwen \
    nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3 sleep 864000
```
- 如果是自己编译的镜像
```bash
docker run -d \
    --name triton \
    --net host \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v ${PWD}/tensorrtllm_backend:/tensorrtllm_backend \
    -v ${PWD}/Qwen-TensorRT-LLM/examples/qwen:/root/qwen \
    triton_trt_llm:v0.7.0 sleep 864000
```
6. 检查服务
- 进入容器
```bash
docker exec -it triton /bin/bash
```
- 检查英伟达驱动
```bash
nvidia-smi
```
- 检查tritonserver版本，至少和上面提到的一样，是2.39（自己编译的容器是2.39）
```bash
cat /opt/tritonserver/TRITON_VERSION
```
- 检查tensorrtllm_backend版本，该数值必须和官方github仓库的0.7.0版本的tool/version.txt文件内容一致，[官方仓库链接](https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.7.0/tools/version.txt)
```bash
cat /tensorrtllm_backend/tools/version.txt
```
7. 安装git-lfs，然后克隆代码（如果是自己编译的容器，这步可以省略）
```bash
apt update
apt install git-lfs
```
8. 安装pytorch2.1.0，目前还不支持2.2。
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
9. 直接通过pip安装TensorRT-LLM （如果是自己编译的容器，这步可以省略）
```bash
pip install tensorrt_llm==0.7.0 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```


### 编译Engine
- 参考我项目的[readme](https://github.com/Tlntin/Qwen-TensorRT-LLM)
1. 进入容器
```bash
docker exec -it triton /bin/bash
```

2. 重复之前的操作，安装qwen的依赖，编译Engine,推荐开启inflight-batching+smooth int8，参考命令
- 进入qwen目录
```bash
cd /root/qwen
```
- 安装依赖
```bash
pip install -r requirements.txt
```
- 转smooth int8 权重
```bash
python3 hf_qwen_convert.py --smoothquant=0.5
```
- 编译(防止显存不足，暂时将最大输入输出设置为2048)
```bash
python3 build.py \
	--use_smooth_quant \
	--per_token \
	--per_channel \
	--use_inflight_batching \
	--paged_kv_cache \
	--remove_input_padding \
	--max_input_len 2048 \
	--max_new_tokens 2048
```
- 运行一下做个测试
```bash
python3 run.py
```

### 部署Triton
- 参考tensorrtllm_backend 0.7.0的[readme](https://github.com/triton-inference-server/tensorrtllm_backend/tree/v0.7.0)
- 同时参考llama的[详细部署教程](https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.7.0/docs/llama.md)
1. 进入容器
```bash
docker exec -it triton /bin/bash
```
2. 构建好目录
```bash
cd /tensorrtllm_backend
mkdir triton_model_repo
```
3.  复制上一部分编译好的Engine文件
```bash
cd /root/qwen/trt_engines/fp16/1-gpu/
cp -r ./* /tensorrtllm_backend/triton_model_repo/tensorrt_llm/1/
```
4. 复制tokenzer文件
```bash
cd /root/qwen/
cp -r qwen_7b_chat /tensorrtllm_backend/triton_model_repo/tensorrt_llm/

# 删除tokenizer目录的Huggingface模型文件（可选）
rm /tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen_7b_chat/*.safetensors
```
5. 编写Triton中的预处理配置和后处理配置， 修改`triton_model_repo/preprocessing/config.pbtxt`文件和`triton_model_repo/postprocessing/config.pbtxt`文件
- 修改前
```pbtxt
parameters {
  key: "tokenizer_dir"
  value: {
	string_value: "${tokenizer_dir}"
  }
}

parameters {
  key: "tokenizer_type"
  value: {
	string_value: "${tokenizer_type}"
  }
}
```
- 修改后
```pbtxt
parameters {
  key: "tokenizer_dir"
  value: {
	string_value: "/tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen_7b_chat"
  }
}

parameters {
  key: "tokenizer_type"
  value: {
	string_value: "auto"
  }
}
```
6. 简单修改一下preprocess/postprocess的model.py的initialize函数，示例是llama的，我们要改成qwen的tokenizer配置。
- 修改前：
```python
self.tokenizer.pad_token = self.tokenizer.eos_token
self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token,
                                            add_special_tokens=False)[0]
```
- 修改后
```bash
import os


gen_config_path = os.path.join(tokenizer_dir, 'generation_config.json')
with open(gen_config_path, 'r') as f:
    gen_config = json.load(f)
chat_format = gen_config['chat_format']
if chat_format == "raw":
    self.eos_id = gen_config['eos_token_id']
    self.pad_id = gen_config['pad_token_id']
elif chat_format == "chatml":
    self.pad_id = self.eos_id = self.tokenizer.im_end_id
else:
    raise Exception("unkown chat format ", chat_format)
eos_token = self.tokenizer.decode(self.eos_id)
self.tokenizer.eos_token = self.tokenizer.pad_token = eos_token
```
7. 然后，参考tensorrtllm_backend 0.7.0的[readme](https://github.com/triton-inference-server/tensorrtllm_backend/tree/v0.7.0)，将表格里面的变量填好，比如batch_size,是否开启流等，每个版本略有不同，可以自行斟酌，此处不再过多论述。（注：`triton_max_batch_size`最低应为4；`decoupled_mode`建议设置为true；`gpt_model_path`设置为Engine的路径，也就是/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1；`gpt_model_type`设置为V1用于开启inflight-batching；`preprocessing_instance_count`和`postprocessing_instance_count`为分词的时候用多少个CPU核心，可以设置为你的CPU核心数；`max_queue_delay_microseconds`队列最大延迟微秒可以设置为1000，这个参数貌似是间隔多久才返回请求给客户端的；`bls_instance_count`同样可以根据cpu核心数设置。`exclude_input_in_output`设置为true,也就是返回时排除输入。）
8. 启动服务
```bash
cd /tensorrtllm_backend
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_backend/triton_model_repo
```
9. 另外开一个终端，测试一下http效果。
- 请求
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate \
-d '{"text_input": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，你叫什么？<|im_end|>\n<|im_start|>assistant\n", "max_tokens": 100, "bad_words": "", "stop_words": "", "end_id": [151645], "pad_id": [151645]}'

```
- 输出结果
```text
{"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"你好，我是来自阿里云的大规模语言模型，我叫通义千问。"}%    
```

### 调用服务
###### python客户端请求
1. 安装python依赖（可选）
```bash
pip install tritonclient transformers gevent geventhttpclient tiktoken grpcio
```
2. 运行`qwen/triton_client/inflight_batcher_llm_client.py`文件即可开启
```bash
cd /root/qwen/triton_client
python3 inflight_batcher_llm_client.py
```

3. 测试结果
```bash
====================
Human: 你好
Output: 你好！有什么我可以帮助你的吗？
Human: 你叫什么？
Output: 我是来自阿里云的大规模语言模型，我叫通义千问。
```

##### http流式调用
1. 前提
- 编译的Engine开启了`paged_kv_cache`
- 部署triton时，`tensorrt_llm/config.pbtxt`里面的`gpt_model_type`对应的value为inflight_batching
2. 运行命令
```bash
curl -X POST localhost:8000/v2/models/ensemble/generate_stream \
-d '{"text_input": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，你叫什么？<|im_end|>\n<|im_start|>assistant\n", "max_tokens": 100, "bad_words": "", "stop_words": "", "end_id": [151645], "pad_id": [151645], "stream": true}'
```
3. 输出结果：
```bash
data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":0.0,"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"你好"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"，"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"我是"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"来自"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"阿里"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"云"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"的大"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"规模"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"语言"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"模型"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"，"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"我"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"叫"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"通"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"义"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"千"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"问"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":"。"}

data: {"cum_log_probs":0.0,"model_name":"ensemble","model_version":"1","output_log_probs":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"sequence_end":false,"sequence_id":0,"sequence_start":false,"text_output":""}

```

### 关闭triton服务
```bash
pkill tritonserver
```