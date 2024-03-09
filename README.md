### [README FOR ENGLISH](qwen/README.md)
# 总述
### 背景介绍
- 介绍本工作是 <a href="https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023">NVIDIA TensorRT Hackathon 2023</a> 的参赛题目，本项目使用TRT-LLM完成对Qwen-7B-Chat实现推理加速。相关代码已经放在[release/0.1.0](https://github.com/Tlntin/Qwen-TensorRT-LLM/tree/release/0.1.0)分支，感兴趣的同学可以去该分支学习完整流程。
- 本项目[release/0.5.0](https://github.com/Tlntin/Qwen-TensorRT-LLM/tree/release/0.5.0)分支和TensorRT-LLM官方仓库[release/0.5.0](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0)分支对齐，所有功能均在该分支上面进行测试。
- main分支目前和TensorRT-LLM官方仓库[v0.7.0](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.7.0)对齐，该版本已经支持Qwen，但是可能支持的功能特性还有不足，故我们决定继续更新该仓库。

### 功能概述

- FP16 / BF16(实验性)
- INT8 Weight-Only & INT8 Smooth Quant & INT4 Weight-Only & INT4-AWQ & INT4-GPTQ
- INT8 KV CACHE
- Tensor Parallel（多卡并行）
- 基于gradio搭建web demo
- 支持triton部署api，结合`inflight_batching`实现最大吞吐/并发。
- 支持fastapi搭建兼容openai请求的api，并且支持function call调用。
- 支持cli命令行对话。
- 支持langchain接入。

### 支持的模型（qwen/qwen1.5/qwen-vl)

- base模型（实验性）：[Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B)、[Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)、[Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)、[Qwen-72B](https://huggingface.co/Qwen/Qwen-72B)（实验性）、[QWen-VL](https://huggingface.co/Qwen/Qwen-VL)
- chat模型（推荐）：[Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)、[Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)、[Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)、[Qwen-72B-Chat](https://huggingface.co/Qwen/Qwen-72B-Chat)（实验性）、[QWen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- chat-int4模型：[Qwen-1_8B-Chat-Int4](https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4)、[Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4)、[Qwen-14B-Chat-Int4](https://huggingface.co/Qwen/Qwen-14B-Chat-Int4)、、[Qwen-72B-Chat-Int4](https://huggingface.co/Qwen/Qwen-72B-Chat-Int4)（实验性）、[Qwen-VL-Chat-Int4](https://huggingface.co/Qwen/Qwen-VL-Chat-Int4)

### 相关教程：
- 本项目配套B站教程：

  <a href="https://www.bilibili.com/video/BV12M411D7uS/"><img src="https://s2.loli.net/2024/03/09/mO47upAkFgyVn6o.png" alt="bilibili"></a>

- 本项目配套博客适配概述：[如何在 NVIDIA TensorRT-LLM 中支持 Qwen 模型](https://developer.nvidia.com/zh-cn/blog/qwen-model-support-nvidia-tensorrt-llm)

- [TensorRT-LLM的模型量化：实现与性能科普视频](https://www.bilibili.com/video/BV1Pw411h7nM/?spm=a2c22.12281976.0.0.6ee62084utHBCm)

- [Triton23.12部署TensorRT-LLM,实现http查询](./docs/triton_deploy_trt-llm.md)



### 软硬件要求

- Linux最佳，已安装docker，并且安装了nvidia-docker（[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)），Windows理论也可以，但是还未测试，感兴趣可以自己研究一下。
- Windows参考这个教程：[链接](https://zhuanlan.zhihu.com/p/676095460)
- 有英伟达显卡（30系，40系，V100/A100等），以及一定的显存、内存、磁盘。结合[Qwen官方推理要求](https://github.com/QwenLM/Qwen/blob/main/README_CN.md#%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD)，预估出下面的要求，详见表格（仅编译期最大要求），仅供参考：

<table>
    <tr>
        <td>Model Size</td>
        <td>Quantization</td>
        <td>GPU Memory Usage (GB)</td>
        <td>CPU Memory Usage (GB)</td>
        <td>Disk Usage (GB)</td>
    </tr>
    <!-- 1.8b -->
    <tr>
        <td rowspan="7">1.8B</td>
        <td>fp16</td>
        <td>5</td>
        <td>15</td>
        <td>11</td>
    </tr>
    <tr>
        <td>int8 smooth quant</td>
        <td>5</td>
        <td>15</td>
        <td>22</td>
    </tr>
    <tr>
        <td>int8 weight only</td>
        <td>4</td>
        <td>12</td>
        <td>9</td>
    </tr>
    <tr>
        <td>int4 weight only</td>
        <td>4</td>
        <td>10</td>
        <td>7</td>
    </tr>
    <tr>
        <td>int4 gptq (raw)</td>
        <td>4</td>
        <td>10</td>
        <td>6</td>
    </tr>
    <tr>
        <td>int4 gptq (manual)</td>
        <td>5</td>
        <td>13</td>
        <td>14</td>
    </tr>
    <tr>
        <td>int4 awq</td>
        <td>5</td>
        <td>13</td>
        <td>18</td>
    </tr>
    <!-- 7b -->
    <tr>
        <td rowspan="7">7B</td>
        <td>fp16</td>
        <td>21</td>
        <td>59</td>
        <td>42</td>
    </tr>
    <tr>
        <td>int8 smooth quant</td>
        <td>21</td>
        <td>59</td>
        <td>84</td>
    </tr>
    <tr>
        <td>int8 weight only</td>
        <td>14</td>
        <td>39</td>
        <td>28</td>
    </tr>
    <tr>
        <td>int4 weight only</td>
        <td>10</td>
        <td>29</td>
        <td>21</td>
    </tr>
    <tr>
        <td>int4 gptq (raw)</td>
        <td>10</td>
        <td>29</td>
        <td>16</td>
    </tr>
    <tr>
        <td>int4 gptq (manual)</td>
        <td>21</td>
        <td>51</td>
        <td>42</td>
    </tr>
    <tr>
        <td>int4 awq</td>
        <td>21</td>
        <td>51</td>
        <td>56</td>
    </tr>
    <!-- 14b -->
    <tr>
        <td rowspan="7">14B</td>
        <td>fp16</td>
        <td>38</td>
        <td>106</td>
        <td>75</td>
    </tr>
    <tr>
        <td>int8 smooth quant</td>
        <td>38</td>
        <td>106</td>
        <td>150</td>
    </tr>
    <tr>
        <td>int8 weight only</td>
        <td>24</td>
        <td>66</td>
        <td>47</td>
    </tr>
    <tr>
        <td>int4 weight only</td>
        <td>16</td>
        <td>46</td>
        <td>33</td>
    </tr>
    <tr>
        <td>int4 gptq (raw)</td>
        <td>16</td>
        <td>46</td>
        <td>26</td>
    </tr>
    <tr>
        <td>int4 gptq (manual)</td>
        <td>38</td>
        <td>90</td>
        <td>66</td>
    </tr>
    <tr>
        <td>int4 awq</td>
        <td>38</td>
        <td>90</td>
        <td>94</td>
    </tr>
    <!-- 72b -->
    <tr>
        <td rowspan="7">72B</td>
        <td>fp16</td>
        <td>181</td>
        <td>506</td>
        <td>362</td>
    </tr>
    <tr>
        <td>int8 smooth quant</td>
        <td>181</td>
        <td>506</td>
        <td>724</td>
    </tr>
    <tr>
        <td>int8 weight only</td>
        <td>102</td>
        <td>284</td>
        <td>203</td>
    </tr>
    <tr>
        <td>int4 weight only</td>
        <td>61</td>
        <td>171</td>
        <td>122</td>
    </tr>
    <tr>
        <td>int4 gptq (raw)</td>
        <td>61</td>
        <td>171</td>
        <td>98</td>
    </tr>
    <tr>
        <td>int4 gptq (manual)</td>
        <td>181</td>
        <td>434</td>
        <td>244</td>
    </tr>
    <tr>
        <td>int4 awq</td>
        <td>181</td>
        <td>434</td>
        <td>406</td>
    </tr>
</table>



# 运行指南

### 准备工作
1. 下载镜像。
    - 官方triton镜像23.12，对应TensorRT-LLM版本为0.7.0，不含TensorRT-LLM开发包。
      ```bash
      docker pull nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3
      docker tag nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3 tensorrt_llm/release
      ```
      
    - 安装pytorch2.1.0，目前还不支持2.2。

      ```bash
       pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
      ```
    - 使用pip直接安装官方编译好的tensorrt_llm
      ```bash
      pip install tensorrt_llm==0.7.0 --extra-index-url https://pypi.nvidia.com --extra-index-url https://download.pytorch.org/whl/cu121
      ```
    
    - AutoDL镜像，不含triton，版本为0.6.1，无卡用户可以体验玩玩，[链接](https://www.codewithgpu.com/i/NVIDIA/TensorRT-LLM/tensorrt_llm)
    
      
    
2. 拉取本项目代码

    ```bash
    git clone https://github.com/Tlntin/Qwen-TensorRT-LLM.git
    cd Qwen-TensorRT-LLM
    ```

3. 进入项目目录，然后创建并启动容器，同时将本地`qwen`代码路径映射到`/app/tensorrt_llm/examples/qwen`路径，然后打开8000和7860端口的映射，方便调试api和web界面。

    ```bash
    docker run --gpus all \
      --name trt_llm \
      -d \
      --ipc=host \
      --ulimit memlock=-1 \
      --restart=always \
      --ulimit stack=67108864 \
      -p 8000:8000 \
      -p 7860:7860 \
      -v ${PWD}/examples/qwen:/app/tensorrt_llm/examples/qwen \
      tensorrt_llm/release sleep 8640000
    ```

4. 进入docker容器里面的qwen路径，安装提供的Python依赖

    ```bash
    cd /app/tensorrt_llm/examples/qwen/
    pip install -r requirements.txt
    ```

5. 下载模型，例如`QWen-7B-Chat`模型，然后将文件夹重命名为`qwen_7b_chat`，最后放到`qwen/`路径下即可。

6. 修改编译参数（可选）

    - 默认编译参数，包括batch_size, max_input_len, max_new_tokens, seq_length都存放在`default_config.py`中
    - 默认模型路径，包括`hf_model_dir`（模型路径）和`tokenizer_dir`（分词器路径）以及`int4_gptq_model_dir`（手动gptq量化输出路径），可以改成你自定义的路径。
    - 对于24G显存用户，直接编译即可，默认是fp16数据类型，max_batch_size=2
    - 对于低显存用户，可以降低max_batch_size=1，或者继续降低max_input_len, max_new_tokens
    - 默认的seq_lenght是2048，有的模型是8192有的是2048，一般看`config.json`里面这个参数对应的数值是啥，如果没改对后面会警告或者报错提醒一下（准备加入这个功能）。

### 运行指南（fp16模型）
1. 编译。

    - 编译fp16（注：`--remove_input_padding`和`--enable_context_fmha`为可选参数，可以一定程度上节省显存）。

    ```bash
    python3 build.py --remove_input_padding --enable_context_fmha
    ```
    
    - 编译 int8 (weight only)。

    ```bash
    python3 build.py --use_weight_only --weight_only_precision=int8
    ```
    
    - 编译int4 (weight only)
    ```bash
    python3 build.py --use_weight_only --weight_only_precision=int4
    ```
    
    - 对于如果单卡装不下，又不想用int4/int8量化，可以选择尝试tp = 2，即启用两张GPU进行编译 （注：tp功能目前只支持从Huggingface格式构建engine）
    ```bash
    python3 build.py --world_size 2 --tp_size 2
    ```
    
2. 运行。编译完后，再试跑一下，输出`Output: "您好，我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>"`这说明成功。

    - tp = 1（默认单GPU）时使用python直接运行run.py
    ```bash
    python3 run.py
    ```

    - tp = 2（2卡用户，或者更多GPU卡）时，使用`mpirun`命令来运行run.py
    ```bash
    mpirun -n 2 --allow-run-as-root python run.py
    ```

3. 验证模型精度。可以试试跑一下`summarize.py`，对比一下huggingface和trt-llm的rouge得分。对于`网络不好`的用户，可以从网盘下载数据集，然后按照使用说明操作即可。

     - 百度网盘：链接: https://pan.baidu.com/s/1UQ01fBBELesQLMF4gP0vcg?pwd=b62q 提取码: b62q 
     - 谷歌云盘：https://drive.google.com/drive/folders/1YrSv1NNhqihPhCh6JYcz7aAR5DAuO5gU?usp=sharing
     - 跑hugggingface版

     ```bash
     python3 summarize.py --backend=hf
     ```

     - 跑trt-llm版

     ```bash
     python3 summarize.py --backend=trt_llm
     ```

     - 注：如果用了网盘的数据集，解压后加载就需要多两个环境变量了，运行示范如下：

     ```bash
     HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 summarize.py --backend=hf
     或者
     HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 summarize.py --backend=trt_llm
     ```

     - 一般来说，如果trt-llm的rouge分数和huggingface差不多，略低一些（1以内）或者略高一些（2以内），则说明精度基本对齐。

4. 测量模型吞吐速度和生成速度。需要下载`ShareGPT_V3_unfiltered_cleaned_split.json`这个文件。

     - 可以通过wget/浏览器直接下载，[下载链接](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)
     - 也可通过百度网盘下载，链接: https://pan.baidu.com/s/12rot0Lc0hc9oCb7GxBS6Ng?pwd=jps5 提取码: jps5
     - 下载后同样放到`examples/qwen/`路径下即可
     - 测量前，如果需要改max_input_length/max_new_tokens，可以直接改`default_config.py`即可。一般不推荐修改，如果修改了这个，则需要重新编译一次trt-llm，保证两者输入数据集长度统一。
     - 测量huggingface模型

     ```bash
     python3 benchmark.py --backend=hf --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --hf_max_batch_size=1
     ```

     - 测量trt-llm模型 (注意：`--trt_max_batch_size`不应该超过build时候定义的最大batch_size，否则会出现内存错误。)

     ```bash
     python3 benchmark.py --backend=trt_llm --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --trt_max_batch_size=1
     ```

### 运行指南（Smooth Quant篇）
1. 注意：运行Smooth Quant需要将huggingface模型完全加载到GPU里面，用于构建int8标定数据集，所以需要提前确保你的显存够大，能够完全加载整个模型。

2. 将Huggingface格式的数据转成FT(FastTransformer)需要的数据格式
    ```bash
    python3 hf_qwen_convert.py --smoothquant=0.5
    ```

3. 开始编译trt_engine
    - 普通版
    ```bash
    python3 build.py --use_smooth_quant
    ```

    - 升级版（理论上运行速度更快，推理效果更好，强烈推荐）
    ```bash
    python3 build.py --use_smooth_quant --per_token --per_channel
    ```
4. 编译完成，run/summarize/benchmark等等都和上面的是一样的了。

### 运行指南（int8-kv-cache篇）
1. 注意：运行int8-kv-cache需要将huggingface模型完全加载到GPU里面，用于构建int8标定数据集，所以需要提前确保你的显存够大，能够完全加载整个模型。

2. 将Huggingface格式的数据转成FT(FastTransformer)需要的数据格式。
```bash
python3 hf_qwen_convert.py --calibrate-kv-cache
```

3. 编译int8 weight only + int8-kv-cache
```bash
python3 build.py --use_weight_only --weight_only_precision=int8 --int8_kv_cache
```

### 运行指南（int4-gptq篇）
1. 需要安装[auto-gptq](https://github.com/PanQiWei/AutoGPTQ)模块，并且升级transformers模块版本到最新版（建议optimum和transformers都用最新版，否则可能有乱码问题），参考[issue/68](https://github.com/Tlntin/Qwen-TensorRT-LLM/issues/68)。（注：安装完模块后可能会提示tensorrt_llm与其他模块版本不兼容，可以忽略该警告）
```bash
pip install auto-gptq optimum
pip install transformers -U
```

2. 手动获取标定权重（可选）
- 转权重获取scale相关信息，默认使用GPU进行校准，需要能够完整加载模型。（注：对于Qwen-7B-Chat V1.0，可以加上`--device=cpu`来尝试用cpu标定，但是时间会很长）
```bash
python3 gptq_convert.py
```
- 编译TensorRT-LLM Engine
```bash
python build.py --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group
```
- 如果想要节省显存（注：只能用于单batch），可以试试加上这俩参数来编译Engine
```bash
python build.py --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --remove_input_padding \
                --enable_context_fmha
```

3. 使用官方int4权重，例如Qwen-xx-Chat-Int4模型（推荐）
- 编译模型，注意设置hf模型路径和`--quant_ckpt_path`量化后权重路径均设置为同一个路径，下面是1.8b模型的示例（其他模型也是一样操作）
```bash
python build.py --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --hf_model_dir Qwen-1_8B-Chat-Int4 \
                --quant_ckpt_path Qwen-1_8B-Chat-Int4
```
- 运行模型，这里需要指定一下tokenizer路径
```bash
python3 run.py --tokenizer_dir=Qwen-1_8B-Chat-Int4
```

### 运行指南（int4-awq篇）
1. 需要下载并安装nvidia-ammo模块
```bash
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-ammo~=0.5.0
```
2. 修改ammo代码，加上qwen支持（不加上会报错），下面是一个简单的参考案例：
- 先在vscode，任意写一个python文件，导入下面的函数
```python
from tensorrt_llm.models.quantized.ammo import quantize_and_export
```
- 然后contrl + 鼠标左按键，单击`quantize_and_export`函数，查看它的内部实现。
- model_lookup中，加上下面这段代码，用来支持Qwen
```bash
    ("qwen", ): "qwen",
```
- 修改后长这样：
```bash
model_lookup = {
    ("llama", "mistral"): "llama",
    ("gptj", ): "gptj",
    ("falcon", "rw"): "falcon",
    ("baichuan", ): "baichuan",
    ("mpt", ): "mpt",
    ("gpt2", ): "gpt2",
    ("chatglm", ): "chatglm",
    ("qwen", ): "qwen",
}
```
- before
```bash
if export_path:
    with torch.inference_mode():
        export_model_config(
            model,
            model_type,
            torch.float16,
            export_dir=export_path,
            inference_tensor_parallel=tensor_parallel_size,
        )
    logger.info(f"Quantized model exported to :{export_path}")
```
- after
```bash
if export_path:
    with torch.inference_mode():
        if qformat == "int4_awq" and model_type == "qwen":
            torch.save(model.state_dict(), export_path)
        else:
            export_model_config(
                model,
                model_type,
                torch.float16,
                quantization=qformat,
                export_dir=export_path,
                inference_tensor_parallel=tensor_parallel_size,
            )
    logger.info(f"Quantized model exported to :{export_path}")
```
3. 运行int4-awq量化代码，导出校准权重。
```bash
python3 quantize.py --export_path ./qwen_7b_4bit_gs128_awq.pt
```
4. 运行build.py，用于构建TensorRT-LLM Engine。
```bash
python build.py --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --quant_ckpt_path ./qwen_7b_4bit_gs128_awq.pt
```
5. 如果想要节省显存（注：只能用于单batch），可以试试加上这俩参数来编译Engine
```bash
python build.py --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --remove_input_padding \
                --enable_context_fmha \
                --quant_ckpt_path ./qwen_7b_4bit_gs128_awq.pt
```

### 其他应用
1. 尝试终端对话。运行下面的命令，然后输入你的问题，直接回车即可。

     ```bash
     python3 cli_chat.py
     ```

2. 部署api，并调用api进行对话。

      - 部署api

      ```bash
      python3 api.py
      ```

      - 另开一个终端，进入`qwen/client`目录，里面有4个文件，分别代表不同的调用方式。
      - `async_client.py`，通过异步的方式调用api，通过SSE协议来支持流式输出。
      - `normal_client.py`，通过同步的方式调用api，为常规的HTTP协议，Post请求，不支持流式输出，请求一次需要等模型生成完所有文字后，才能返回。
      - `openai_normal_client.py`，通过`openai`模块直接调用自己部署的api，该示例为非流式调用，请求一次需要等模型生成完所有文字后，才能返回。。
      - `openai_stream_client.py`，通过`openai`模块直接调用自己部署的api，该示例为流式调用。
      - 注意：需要`pydantic`模块版本>=2.3.2，否则将会出现`ChatCompletionResponse' object has no attribute 'model_dump_json'`报错，参考[issue](https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM/issues/27)

3. 尝试网页对话（可选，需要先部署api）。运行下面的命令，然后打开本地浏览器，访问：[http://127.0.0.1:7860](http://127.0.0.1:7860) 即可

     ```bash
     python3 web_demo.py
     ```
     - 默认配置的web_demo.py如下：
     ```python
     demo.queue().launch(share=False, inbrowser=True)
     ```
     - 如果是服务器运行，建议改成这样
     ```python
     demo.queue().launch(server_name="0.0.0.0", share=False, inbrowser=False) 
     ```
     - web_demo参数说明
         - `share=True`: 代表将网站穿透到公网，会自动用一个随机的临时公网域名，有效期3天，不过这个选项可能不太安全，有可能造成服务器被攻击，不建议打开。
         - `inbrowser=True`: 部署服务后，自动打开浏览器，如果是本机，可以打开。如果是服务器，不建议打开，因为服务器也没有谷歌浏览器给你打开。
         - `server_name="0.0.0.0"`: 允许任意ip访问，适合服务器，然后你只需要输入`http://[你的ip]: 7860`就能看到网页了，如果不开这个选择，默认只能部署的那台机器才能访问。
         - `share=False`：仅局域网/或者公网ip访问，不会生成公网域名。
         - `inbrowser=False`： 部署后不打开浏览器，适合服务器。

4. web_demo运行效果（测试平台：RTX 4080, qwen-7b-chat, int4 weight only)

https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM/assets/28218658/940c1ed1-14f7-45f6-bf13-67c8f289c956
