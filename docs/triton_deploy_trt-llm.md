# Triton24.02 部署TensorRT-LLM,实现http查询

### 选择正确的环境

1. 选择版本。查询nvidia[官方文档](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)，可以看到目前最新的容器是24.02。

- 在**NVIDIA Driver**这一行，它推荐的英伟达驱动版本是545以上，对于数据卡，可以适当降低。如果你是游戏卡，驱动版本没有545，也不想升级，那么建议至少不要低太多，比如535其实也可以。
  ![38a9563ae5435516a18043d93494b7eb.png](https://s2.loli.net/2024/04/19/lBrwuzbgJ725yNS.png)
- 在**Triton Inference Server**这一行，可以看到它内置了triton server版本是2.43，需要的TensorRT-LLM版本是0.8.0。
  ![ed50e1a173903ea931e8103aecbe29fb.png](https://s2.loli.net/2024/04/19/19VN8OTfedyva4S.png)

2. 拉取镜像。进入[Nvidia镜像中心](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags)找到tritonserver的镜像，选择和TensorRT-LLM（简称trtllm）有关的容器，然后拷贝镜像地址，最后使用`docker pull`来拉取该镜像。
   ![9205bd0697f97ed061db52fd39994fa2.png](https://s2.loli.net/2024/04/19/Bq5KW28jApnGHXF.png)

```bash
docker pull nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3
```

3. 拉取TensorRT-LLM的项目。

- 可以选择官方项目，但是注意要是v0.8.0

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git -b v0.8.0
```

- 也可以选择我的项目，目前main分支就是0.8.0，后续可能会打成tag，建议实际访问项目地址，查看是否有0.8.0的tag。

```bash
git clone https://github.com/Tlntin/Qwen-TensorRT-LLM
```

- 下面演示是以我的项目为主，在triton_server上面部署Qwen-1.8B-Chat（毕竟这个模型比较小）

4. 拉取tensorrtllm_backend。这个是用来编排tensorrt-llm服务的，需要和TensorRT-LLM版本一致，这里同样选择0.8.0

```bash
git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.8.0
```

5. 启动tritonserver容器

```bash
docker run -d \
    --name triton \
    --net host \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus all \
    -v ${PWD}/tensorrtllm_backend:/tensorrtllm_backend \
    -v ${PWD}/Qwen-TensorRT-LLM/examples:/root/examples \
    nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3 sleep 864000
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

- 检查tritonserver版本，至少和上面提到的一样，是2.43

```bash
cat /opt/tritonserver/TRITON_VERSION
```

- 检查tensorrtllm_backend版本，该数值必须和官方github仓库的0.8.0版本的tool/version.txt文件内容一致，[官方仓库链接](https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.8.0/tools/version.txt)

```bash
cat /tensorrtllm_backend/tools/version.txt
```

<!-- 7. 安装git-lfs，然后克隆代码（如果是自己编译的容器，这步可以省略）

```bash
apt update
apt install git-lfs
``` -->
7. 直接通过pip安装TensorRT-LLM （如果是自己编译的容器，这步可以省略）
```bash
pip install tensorrt_llm==0.8.0 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
```


### 编译Engine

- 参考我项目的[readme](https://github.com/Tlntin/Qwen-TensorRT-LLM)

1. 进入容器

```bash
docker exec -it triton /bin/bash
```

2. 重复之前的操作，安装qwen的依赖。

- 进入qwen2目录

```bash
cd /root/examples/qwen2
```

- 安装依赖

```bash
pip install -r requirements.txt
```

3. 编译，需要在原来Readme编译的基础上开启paged_kv_cache，方便部署inflight-batching

- 例如fp16之前编译是`python3 build.py`，现在改成下面这个。

```bash
python3 build.py --paged_kv_cache --remove_input_padding
```

- 例如int8-smooth-quant
  - 之前编译命令是
    ```bash
    # 转权重
    python3 hf_qwen_convert.py --smoothquant=0.5
    # 编译
    python3 build.py --use_smooth_quant --per_token --per_channel
    ```
  
  - 现在编译命令是
      ```bash
      # 转权重
      python3 hf_qwen_convert.py --smoothquant=0.5
      # 编译
      python3 build.py --use_smooth_quant --per_token --per_channel --paged_kv_cache --remove_input_padding
      ```

4. 运行一下做个测试

```bash
python3 run.py
```

### 临时部署Triton

- 参考tensorrtllm_backend 0.8.0的[readme](https://github.com/triton-inference-server/tensorrtllm_backend/tree/v0.8.0)
- 同时参考llama的[详细部署教程](https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.8.0/docs/llama.md)

1. （可选）直接复用本项目配置（batch_size=2，input=6144, output=2048），这样就跳过后续的第6，第7步骤，需要在容器外操作，懒人必备，不过还是推荐自己改好一些。
```bash
cp -r Qwen-TensorRT-LLM/triton_model_repo tensorrtllm_backend/
```

2. 进入容器

```bash
docker exec -it triton /bin/bash
```

3. 构建好目录

```bash
cd /tensorrtllm_backend
cp all_models/inflight_batcher_llm/ -r triton_model_repo
```

4.  复制上一部分编译好的Engine文件

```bash
cd /root/examples/qwen2/trt_engines/fp16/1-gpu/
cp -r ./* /tensorrtllm_backend/triton_model_repo/tensorrt_llm/1/
```

5. 复制tokenzer文件

```bash
cd /root/examples/qwen2
mkdir /tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat
cp qwen1.5_7b_chat/*.json /tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat/
# 可选，仅适用于qwen1,因为它的tokenizer是tiktoken格式
cp qwen_7b_chat/*.tiktoken /tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat/
cp qwen_7b_chat/*.py /tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat/
```

6. （可选）编写Triton中的预处理配置和后处理配置， 参考[文档](https://github.com/triton-inference-server/tensorrtllm_backend/blob/v0.8.0/docs/llama.md)

```bash
cd /tensorrtllm_backend
export HF_QWEN_MODEL="/tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat"
export ENGINE_DIR="/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1"
# 设置你的batch_size大小
export MAX_BATCH_SIZE=1
export TOKENIZE_TYPE=auto
# 根据cpu线程数定，一般为batch_size的2倍数或者cpu线程的一半
export INSTANCE_COUNT=2
# 我就一张卡，你可以指定用那些卡，用逗号隔开
export GPU_DEVICE_IDS=0


python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt tokenizer_dir:${HF_QWEN_MODEL},tokenizer_type:${TOKENIZE_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}

python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt tokenizer_dir:${HF_QWEN_MODEL},tokenizer_type:${TOKENIZE_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:False,bls_instance_count:${INSTANCE_COUNT},accumulate_tokens:True

python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE}

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:True,max_beam_width:1,engine_dir:${ENGINE_DIR},exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_batching,max_queue_delay_microseconds:600,gpu_device_ids:${GPU_DEVICE_IDS}
```

7. （可选）简单修改一下preprocess/postprocess的model.py的initialize函数，示例是llama的，我们要改成qwen的tokenizer配置。

- 修改前（preprocessing有三行，postprocessing只有一行）：

```python
self.tokenizer.pad_token = self.tokenizer.eos_token
self.tokenizer_end_id = self.tokenizer.encode(
    self.tokenizer.eos_token, add_special_tokens=False)[0]
self.tokenizer_pad_id = self.tokenizer.encode(
    self.tokenizer.pad_token, add_special_tokens=False)[0]
```

- 修改后

```bash
import os


gen_config_path = os.path.join(tokenizer_dir, 'generation_config.json')
with open(gen_config_path, 'r') as f:
	gen_config = json.load(f)
if isinstance (gen_config["eos_token_id"], list):
	pad_id = end_id = gen_config["eos_token_id"][0]
### if model type is base, run this branch
else:
	pad_id = gen_config["bos_token_id"]
	end_id = gen_config["eos_token_id"]
self.tokenizer_pad_id = pad_id
self.tokenizer_end_id = end_id
eos_token = self.tokenizer.decode(end_id)
self.tokenizer.eos_token = self.tokenizer.pad_token = eos_token
```

8. 启动服务，单卡启动。

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
cd /root/examples/triton_client
python3 inflight_batcher_llm_client.py --tokenizer_dir=/tensorrtllm_backend/triton_model_repo/tensorrt_llm/qwen1.5_7b_chat
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

### 永久部署

1. 在上个容器部署时，我们启动的命令是`python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_backend/triton_model_repo`，经过修改`tensorrtllm_backend/scripts/launch_triton_server.py`文件，倒数第二行增加一个`print("cmd", cmd)`在结尾打印出它真实运行的命令如下：

```bash
["mpirun", "--allow-run-as-root", "-n", "1", "/opt/tritonserver/bin/tritonserver", "--model-repository=/tensorrtllm_backend/triton_model_repo", "--grpc-port=8001", "--http-port=8000", "--metrics-port=8002", "--disable-auto-complete-config", "--backend-config=python,shm-region-prefix-name=prefix0_", ":"]
```

2. 编写一个Dockerfile来启动刚刚的命令，替换原来容器自带的命令。

```DockerFile
FROM nvcr.io/nvidia/tritonserver:24.02-trtllm-python-py3
USER root
# Option
# COPY tensorrtllm_backend /tensorrtllm_backend
WORKDIR /tensorrtllm_backend

CMD ["mpirun", "--allow-run-as-root", "-n", "1", "/opt/tritonserver/bin/tritonserver", "--model-repository=/tensorrtllm_backend/triton_model_repo", "--grpc-port=8001", "--http-port=8000", "--metrics-port=8002", "--disable-auto-complete-config", "--backend-config=python,shm-region-prefix-name=prefix0_", ":"]
```

3. 编译新镜像，命名为tritonserver:24.02

```bash
docker build . -t tritonserver:24.02
```

4. 测试一下是否ok

```bash
docker run -it \
    --name triton_server \
    --net host \
    --shm-size=2g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PWD}/tensorrtllm_backend:/tensorrtllm_backend \
    --gpus all \
    tritonserver:24.02
```

- 测试一下请求，没问题就退出，然后删除该容器

```bash
docker rm -f triton_server
```

5. 永久开启该容器，设置后台启动，并且设置自动重启

```bash
docker run -d \
    --name triton_server \
    --net host \
    --shm-size=2g \
    --restart always \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v ${PWD}/tensorrtllm_backend:/tensorrtllm_backend \
    --gpus all \
    tritonserver:24.02
```

6. 查看一下这个容器运行情况，发现正常。

```bash
docker logs triton_server
```
