### 总述

- 介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，本项目将使用TRT-LLM完成对Qwen-7B-Chat实现推理加速。
- 原始模型：Qwen-7B-Chat
- 原始模型URL：[Qwen-7B-Chat 🤗](https://huggingface.co/Qwen/Qwen-7B-Chat) [Qwen-7B-Chat Github](https://github.com/QwenLM/Qwen-7B) 
- 注：Hugggingface的Qwen-7B-Chat貌似下架了，需要的可以用网盘下载。
    - [百度网盘](https://pan.baidu.com/s/1Ra4mvQcRCbkzkReFYhk3Vw?pwd=6fxh) 提取码: 6fxh 
    - [Mega网盘](https://mega.nz/folder/d3YH2SaJ#QSoyfqSXBmNKlpyro6lvVA)
    - [123pan](https://www.123pan.com/s/oEqDVv-LFik.html) 提取码:JAUb
    - 123pan直链: `https://vip.123pan.cn/1811989133/trt2023%E7%9B%B8%E5%85%B3/qwen_7b_chat.tar.gz`, 下载解压即可

- 选题类型：2+4（注：2指的是TRT-LLM实现新模型。4指的是在新模型上启用了TRT-LLM现有feature）

### 主要贡献

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：
##### 优化效果
（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开

- 精度：fp16 基本和原版一样，int8 / int4(weight only) Rouge分数略有提高。总的来说，和原版基本相差不大。
- 加速比：吞吐加速比最高**4.29**倍，生成加速比最高**5.24**倍。

##### 运行指南

1. 准备工作
   - 有一个英伟达显卡，建议12G显存以上，推荐24G（注：12G显存可以用int4, 16G显存可以用int8, 24G显存可以用fp16）。
   - 需要Linux系统，WSL或许也可以试试。
   - 已经安装了docker，并且安装了nvidia-docker
   - 需要较大的磁盘空间，最少50G以上，推荐100G。
   - 需要较大的CPU内存，最少32G，推荐64G以上。

2. 拉取本项目代码

    ```bash
    git clone https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM.git
    ```

3. 拉取docker镜像。

    ```bash
    docker pull registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1
    ```

4. 进入项目目录，然后创建并启动容器，同时将本地代码路径映射到`/root/workspace/trt2023`路径

    ```bash
    cd Qwen-7B-Chat-TensorRT-LLM

    docker run --gpus all \
      --name trt2023 \
      -d \
      --ipc=host \
      --ulimit memlock=-1 \
      --restart=always \
      --ulimit stack=67108864 \
      -v ${PWD}:/root/workspace/trt2023 \
      registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 sleep 8640000
    ```

5. 下载模型`QWen-7B-Chat`模型（可以参考总述部分），然后将文件夹重命名为`qwen_7b_chat`，最后放到`tensorrt_llm_july-release-v1/examples/qwen/`路径下即可。
6. 安装根目录的提供的Python依赖，然后再进入qwen路径

    ```bash
    pip install -r requirements.txt
    cd tensorrt_llm_july-release-v1/examples/qwen/
    ```

7. 将Huggingface格式的数据转成FT(FastTransformer)需要的数据格式

    ```bash
    python3 hf_qwen_convert.py
    ```

8. 修改编译参数（可选）

    - 默认编译参数，包括batch_size, max_input_len, max_new_tokens都存放在`default_config.py`中
    - 对于24G显存用户，直接编译即可，默认是fp16数据类型，max_batch_size=2
    - 对于低显存用户，可以降低max_batch_size=1，或者继续降低max_input_len, max_new_tokens

9. 开始编译。
   - 对于24G显存用户，可以直接编译fp16。

    ```bash
    python3 build.py
    ```

    - 对于16G显存用户，可以试试int8 (weight only)。

    ```bash
    python3 build.py --use_weight_only --weight_only_precision=int8
    ```

    - 对于12G显存用户，可以试试int4 (weight only)
    ```bash
    python3 build.py --use_weight_only --weight_only_precision=int4
    ```

10. 试运行（可选）编译完后，再试跑一下，输出`Output: "您好，我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>"`这说明成功。

    ```bash
    python3 run.py
    ```

11. 验证模型精度精度（可选）。可以试试跑一下`summarize.py`，对比一下huggingface和trt-llm的rouge得分。对于`网络不好`的用户，可以从网盘下载数据集，然后按照使用说明操作即可。

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

12. 测量模型吞吐速度和生成速度（可选）。需要下载`ShareGPT_V3_unfiltered_cleaned_split.json`这个文件。

    - 可以通过wget/浏览器直接下载，[下载链接](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)
    - 也可通过百度网盘下载，链接: https://pan.baidu.com/s/12rot0Lc0hc9oCb7GxBS6Ng?pwd=jps5 提取码: jps5
    - 下载后同样放到`tensorrt_llm_july-release-v1/examples/qwen/`路径下即可
    - 测量前，如果需要改max_input_length/max_new_tokens，可以直接改`default_config.py`即可。一般不推荐修改，如果修改了这个，则需要重新编译一次trt-llm，保证两者输入数据集长度统一。
    - 测量huggingface模型

    ```bash
    python3 benchmark.py --backend=hf --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --hf_max_batch_size=1
    ```

    - 测量trt-llm模型 (注意：`--trt_max_batch_size`不应该超过build时候定义的最大batch_size，否则会出现内存错误。)

    ```bash
    python3 benchmark.py --backend=trt_llm --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --trt_max_batch_size=1
    ```

13. 尝试终端对话（可选）。运行下面的命令，然后输入你的问题，直接回车即可。

    ```bash
    python3 cli_chat.py
    ```

14. 尝试网页对话（可选）。运行下面的命令，然后打开本地浏览器，访问：[http://127.0.0.1:7860](http://127.0.0.1:7860) 即可

    ```bash
    python3 web_demo.py
    ```

15. 部署api，并调用api进行对话（可选）。

    - 部署api

    ```bash
    python3 api.py
    ```

    - 另开一个终端，进入`tensorrt_llm_july-release-v1/examples/qwen/client`目录，里面有4个文件，分别代表不同的调用方式。
    - `async_client.py`，通过异步的方式调用api，通过SSE协议来支持流式输出。
    - `normal_client.py`，通过同步的方式调用api，为常规的HTTP协议，Post请求，不支持流式输出，请求一次需要等模型生成完所有文字后，才能返回。
    - `openai_normal_client.py`，通过`openai`模块直接调用自己部署的api，该示例为非流式调用，请求一次需要等模型生成完所有文字后，才能返回。。
    - `openai_stream_client.py`，通过`openai`模块直接调用自己部署的api，该示例为流式调用。


##### 运行指南（Smooth Quant篇）
1. 前6节和上面一样，参考上面运行就行。

2. 将Huggingface格式的数据转成FT(FastTransformer)需要的数据格式
    ```bash
    python3 hf_qwen_convert.py --smoothquant=0.5
    ```

3. 编译前需要编译一下整个项目，这样后面才能加载新增的rmsnorm插件。参考该[教程](https://www.http5.cn/index.php/archives/30/)。


4. 开始编译trt_engine(未完待续)
    - 普通版
    ```bash
    python3 build.py --use_smooth_quan
    ```

    - 升级版（理论上速度更快一些）
    ```bash
    python3 build.py --use_smooth_quan --per_token --per_channel
    ```

### 主要开发工作

- 请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

##### 开发工作的难点

1. huggingface转llm-trt比较繁琐。
    - 目前只能肉眼观察已有成功案例，例如参考chatglm/llama, 通过debug huggingface版和观察trt-llm版，了解整体思路。
    - 然后观察qwen和chatglm/llama的差异，拿这两个案例做魔改。
    - 通过对比代码，发现examples下面的chatglm-6b的rope embedding和qwen类似，所以chatglm-6b的rope embedding的trt实现可以作为参考项。
    - 移植时发现,rope提前算好了weights，然后分割成了两个cos_embedding和sin_embedding。为确保该方案可行，于是在huggingface版的qwen中实现了类似结构，对比rope_cos和rope_sim的输出结果，以及对应sum值，发现该操作可行，于是将其移植到了qwen trt-llm中。
    - 不过需要注意的是，qwen的dim和max_position_dim和chatglm-6b不一样，加上chatglm-6b trt-llm的rope的inv_freq做了一定约分，导致看起来比较奇怪，所以后续我们直接使用了的qwen原版的inv_freq计算，以及qwen原版的apply_rotary_pos_emb方法。
    - 整体代码魔改自llama, attention/rope参考了chatglm-6b。

2. 首次运行报显存分配错误。
    - 在其附近插入可用显存和需要分配的显存代码，发现是显存不够, 将max_batch_size从默认的8改成2后解决。

3. fp16下，模型的logits无法对齐。
    - 通过阅读`docs/2023-05-19-how-to-debug.md`文档，基本掌握的debug能力，然后按照代码运行顺序，从外到内debug，找到误差所在层。
    - 首先我们对比了wte和rope输出，基本确定这两个layer没有问题。
    - 然后我们打印了qwen_block的每层输入，其中第一个layer的输入hidden_states正常，后续误差逐步增加，所以初步确定误差在QwenBlock这个类中。
    - 由于attention使用了rope相关计算+gpt attention_layer，这里出问题的可能性较大，于是我们在QwenBlock中的attention计算里面加入调试操作，打印其输入与输出结果，并和pytorch做数值对比（主要对比mean, sum数值）。经对比发现QwenBlock的attention输入sum误差在0.2以内，基本ok，但是其输出误差很大，所以需要进一步定位。
    - 由于QwenAttention中采用了rope相关计算+gpt attention plugin的方式组合而成，而plugin调试相对困难，所以我们需要进一步测试gpt attention plugin的输入输出。若输入正常，输出异常，则gpt attention plugin异常，反之，则可能是plugin之前的layer有问题。
    - 在gpt attention plugin处打印发现输入结果无法对齐，于是逐层对比QwenAttention forward过程，最终定位到下面这段代码输出异常。
    ```bash
    qkv = concat([query, key, value], dim=2)
    qkv = qkv.view(
        concat([shape(qkv, 0),
                shape(qkv, 1),
                self.hidden_size * 3])
    )
    ```
    - 在经过2/3天调试后，发现与concat无瓜，是plugin内部再次计算了一次rope,导致qkv结果异常，将`tensorrt_llm.functional.gpt_attention`输入的`rotary_embedding_dim`设置为0后，该问题得到解决。不过最终输出还是有问题，经过对比发现attention输出已经正常，但是QwenBlock里面的self.mlp输出异常，需要进一步对比。
    - 经对比发现原来的`GateMLP` forward函数中，是对第一个layer输出做了silu激活，而qwen是对第二个layer的输出做silu激活，两者存在区别，所以我们又重新建了一个`QwenMLP`类用来实现原版的计算过程。
    - 经过上述优化，经对比输出的logits平均误差大概在0.002左右，基本完成了精度对齐。

4. trt-llm输出结果和pytorch不一致。
    - 此时整个模型的计算过程已经没有问题，也对比了不同step的输出，都是可以对上的，但是输出的结果和pytorch还是有区别：
    ```bash
    input:
    """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你好，请问你叫什么？<|im_end|>
    <|im_start|>assistant
    """
	
    pytorch output: 
    """
    您好，我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>
    """
	
    trt-llm output: 
    """
    您好，我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>
    <|im_start|>assistant
	
    很高兴为您服务。<|im_end|>
    <|endoftext|> решил купить новый ноутбук, но не могу выбрать между тремя предложениями."
    """
    ```
    - 经过对比发现是因为sampling config没有对齐，观察了pytorch原版的后处理逻辑，发现其将`tokenizer.im_start_id, tokenizer.im_end_id`设置为了end of token，考虑到trt-llm只能设置一个end of token, 而在输出时<|im_end|>先于需要<|im_start|>，所以我们将将`EOS_TOKEN`修改为`tokenizer.im_end_id`对应的数字。并将top-p, top-k设置原pytorch版`generation_config.json`中对应的数字。
    - 改完后我们发现结尾存在大量重复`<|im_end|>`（`PAD`和`EOS_TOKEN`解码对应的内容），这个主要是前期past_key_value赋值的时候是默认给了最长的长度`max_input_length+max_output_length`，我们在debug run.py中发现decode的step并不一定输出最大长度，而是经常中途退出循环。所以我们决定将退出时的step返回，如果没有中途退出就返回最大max_output_length, 这样就可以知道模型真实生成的长度。以最大输入长度+真实生成长度做截断，然后再用tokenizer解码，就可以得到最终输出结果了。
    ```bash
    Input: 
    """
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    你好，请问你叫什么？<|im_end|>
    <|im_start|>assistant
    """
	
    Output
    """
    您好，我是来自达摩院的大规模语言模型，我叫通义千问。<|im_end|>
    """
    ```
	- 此时输出结果和pytorch完全一致。
5. 运行`summarize.py`无输出。
    - 由于我们选择qwen-chat-7b是一个chat模型，无法直接输入一段文本做总结，需要写一个专门的prompt（提示语）来让模型做这个总结的工作。
    - 于是我们将原版的`make_context`移植过来，并设置专门的`system_prompt`让模型根据用户输入直接做总结，这样将原始输入加工后再输出结果，使得模型有了总结能力。


    - 至此，在trt-llm上支持qwen模型的基础工作已经做完

##### 开发中的亮点
1. 完整支持原版的logn和ntk（这俩参数是用于增强模型长输入的生成效果，这里的长输入指的是输入长度大于2048小于8192）。不过由于trt-llm的某些bug，导致输入长度>2048时，实际输出会很短甚至为空，详见[issue](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/90)，所以加上rope放gpt attention plugin里面计算更快，所以我们logn注释掉了。
2. 支持`RotaryEmbedding`，并且在input_len > 2048时开启ntk相关计算。
3. 支持自带的`gpt_attention_plugin`与`gemm_plugin`两个plugin，同时将`layernorm_plugin`魔改成`rmsnorm_plugin`以支持smooth_quant量化技术。
4. 同时支持qwen base和chat模型
5. 支持fp16 / int8 (weight only) / int4 (weight only), 理论上最低只需要8G消费级显卡就能运行。
6. 支持在终端对话和使用gradio构建的网页应用中对话，支持流式输出。
7. 支持fastapi部署，支持sse协议来实现流式输出，同时兼容OpenAI的api请求。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。
1. 跑一下`tensorrt_llm_july-release-v1/examples/gpt`的代码，了解一下trt-llm的基本流程。

2. 读`QWen-7B-Chat`的Readme信息，基本了解它是一个`decoder only模型`，并且模型结构类llama。

3. 将`tensorrt_llm_july-release-v1/examples/llama`复制一份，然后重命名为`tensorrt_llm_july-release-v1/examples/qwen`,将里面的模型带`llama`的全部替换为`qwen`。

4. 运行`build.py`时，发现调用了`weight.py`，而llama项目里面的`weight.py`里面有一个`load_from_meta_llama`函数，函数里面有一个`tensorrt_llm.models.LLaMAForCausalLM`，里面定义了整个trt的模型结构。我们用vscode进入其中，将里面的`LLaMAForCausalLM`复制出来，然后新建了一个mode.py,将内容粘贴进去。同样，将里面的模型带`llama`的全部替换为`qwen`

5. 然后我们就开始改本项目的`weight.py`的`load_from_meta_qwen`函数，将huggingface里面的权重名称和trt的权重名称逐步对齐。然后我们运行了一下build.py，发现可以编译通过，但是运行`run.py`结果不对。

6. 我们通过肉眼观察模型结构，发现trt版少了一个`RotaryEmbedding`，这个该如何实现了？观察了一下`tensorrt_llm_july-release-v1/examples/`目录下面的其他项目，突然发现里面的`chatglm6b`有类似的结构，不过他却分成了`position_embedding_cos`和`position_embedding_sin`两个普通的embedding。我们大概知道它或许是一个新的实现方式，但是功能和原版一样。通过观察，我们发现其rope权重是在weight.py里面用numpy手搓直接导入的。导入代码如下：

   - chatglm6b/hf_chatglm6b_convert.py

   ```python
   nMaxSL = 2048
   inv_freq = 10**(-1 / 16 * np.arange(0, 64, 2, dtype=np.float32))
   valueTable = np.matmul(
       np.arange(nMaxSL, dtype=np.float32).reshape(-1, 1),
       np.concatenate([inv_freq, inv_freq],
                      axis=0).reshape(1, -1)).reshape(nMaxSL,
                                                      len(inv_freq) * 2)
   np.cos(valueTable).astype(storage_type).tofile(saved_dir /
                                                  "model.cosTable.weight.bin")
   np.sin(valueTable).astype(storage_type).tofile(saved_dir /
                                                  "model.sinTable.weight.bin")
   print("Save model.cosTable.weight.bin")
   print("Save model.sinTable.weight.bin")
   ```

   - chatglm6b/weight.py

   ```bash
   chatglm6bModel.position_embedding_cos.weight.value = (fromfile(
       dir_path, 'model.cosTable.weight.bin',
       [n_positions, n_embd // n_head // 2]))
   chatglm6bModel.position_embedding_sin.weight.value = (fromfile(
       dir_path, 'model.sinTable.weight.bin',
       [n_positions, n_embd // n_head // 2]))
   ```

   - 经过和原版ChatGLM-6b进行对比，大概知道了2048是原版的最大输入长度，64是他的per_head_dim//2
   - 而qwen里面的apply_rope时的计算和chatglm略有差异，并且qwen的最大输入长度是8192，这些需要略作修改。
   - 修改完后利用debug_mode将对应的数值打印出来，与原版进行对比，发现基本一致。

7. 再次通过通过肉眼观察模型结构，发现原模型Attention中还有logn以及ntk的计算，再增加这两结构后，再次做了一次编译，可以正常编译。

8. 然后就是做fp16对齐了，参考`开发工作的难点`中关于`fp16下，模型的logits无法对齐`部分即可。

9. 在fp16对齐，并且run.py/summarize.py都正常后，我们就开始尝试实现weight only in8/int4，我们先直接运行 `--use_weight_only --weight_only_precision=int8`，编译成功后，再次运行run.py/summarize.py，发现都正常。int4也重复该操作，发现还是正常，说明weight only in8/int4适配成功。

10. 适配完后，发现还有一个smooth_quant量化，可以降低显存，提高推理速度。不过问了导师，说目前只有gpt适配了，后续可能适配bloom和llama，所以我们参考了gpt的代码，新增了一个`hf_qwen_convert.py`文件，用于将huggingface的权重导出到FT(FastTransformer)格式的文件。同时我们将gpt/weight.py里面的`load_from_ft`拷贝到qwen/weight.py，并根据我们导出的权重文件修改进行了简单修改，然后再将build.py里面默认加载函数从`load_from_hf_qwen`换成 `load_from_ft`,不过当开发者没有导出FT权重的时候，还是会自动加载`load_from_hf_qwen`来生成engine。

11. 在用了新的`load_from_ft`方法后，我们又运行了一次run.py和summarize.py，发现输出结果异常，经过对examples/gpt进行调试，发现qwen的attention权重和gpt的attention权重shape顺序略有不同。为了更好的对比差异，我们将`load_from_ft`的加载权重的代码粘贴到`load_from_hf_qwen`，然后一个个变量对比其shape以及value值，对比发现`load_from_hf_qwen`中，非weight_only量化，则权重直接赋值，否则需要多一个转置操作。经过一轮修改，`load_from_ft`的fp16终于正常。然后我们又顺便编译了weight_only的int8/int4，发现也同样正常。

12. 回到smooth quant量化这里，参考example/gpt的smooth quant过程，我们在`hf_qwen_convert.py`里面同样加了`--smoothquant`选项。通过调试`example/gpt/hf_gpt_convert.py`文件，观察它的`smooth_gpt_model`函数的计算方法以及参数的shape，不过他它mlp只有一个fc1, 而我们的mlp有w1/w2和`c_proj`。并且gpt里面有`from smoothquant import capture_activation_range, smooth_gemm`和`from utils.convert import split_and_save_weight`，通过调试这些文件，我们写了自己的对应的函数，并且成功导出了和smooth quant密切相关的int8权重。

13. 再次观察example/gpt的smooth quant过程，参考其build.py文件，发现里面有一个`from tensorrt_llm.models import smooth_quantize`过程，这个函数会将trt-llm原本的模型结构给替换掉，主要替换了layer_norm, attention, 和mlp部分。其中attention基本和我们的qwen一样，只是我们比他多了logn/apply rope/ntk三个，mlp也可以通过简单修改实现。但是关于gpt的layer_norm，我们用的是`RmsNorm`，所以这个部分的smooth quant,需要我们自己实现。

14. RmsNorm相对LayerNorm来说，就是少了一个减均值操作，并且没有bias，所以我们先拷贝了一份layerNorm的kernel。

    - 拷贝操作。

    ```bash
    cd tensorrt_llm_july-release-v1/cpp/tensorrt_llm/kernels
    cp layernormKernels.cu rmsnormKernels.cu
    cp layernormKernels.h rmsnormKernels.h
    ```

    - 拷贝完后我们将里面的mean/bias去掉，并将layernorm关键词换成rmsnorm的关键词。同时我们发现layerNorm的.cu文件里面有个`USE_DIFF_OF_SQUARES`，这个值为Ture时是算方差，为False时算平均值。由于我们不需要平均值，所以这个数值常为True，所以我们将为True的部分保留，为Fasle的部位删除，并且删除了这个变量。
    - 同理我们对plugins目录下面的layerNorm做了同样的操作。

    ```bash
    cd tensorrt_llm_july-release-v1/cpp/tensorrt_llm/plugins
    cp layernormPlugin/* rmsnormPlugin/
    cp layernormQuantizationPlugin/* rmsnormQuantizationPlugin/
    ```

    - 同样和上面一样做替换，去bias，去mean，去`USE_DIFF_OF_SQUARES`（去除这个需要忽略大小写，有的是小写）

15. 参考[教程](https://www.http5.cn/index.php/archives/30/)，我们重新编译了TensorRT-LLM，并且加载了`RmsnormQuantization`插件，但是加载编译后报了一个和cublas相关的错误，错误提示：`terminate called after throwing an instance of 'std::runtime_error'  what():  [TensorRT-LLM Error][int8gemm Runner] Failed to initialize cutlass int8 gemm. Error: Error Internal`。通过多次排查，以及群友的提醒，我们发现这个不是我们plugin的问题，而是smooth_quant_gemm这个插件的问题。该问题可以通过下面的单元测试复现。

    ```bash
    cd tensorrt_llm_july-release-v1/tests/quantization
    python -m unittest test_smooth_quant_gemm.py TestSmoothQuantGemm.test_matmul
    ```
16. 通过[参考教程](https://www.http5.cn/index.php/archives/41/)我们重新编译了Debug版的TRT-LLM，并且可以在vscode中调试TRT-LLM代码。在debug中，我们发现了个别较大的shape存在共享显存分配过多而导致cublas初始化失败的问题。通过增加try/catch功能，我们跳过了失败的策略。然后再次跑test发现出现了下面的错误：
    ```bash
    terminate called after throwing an instance of 'std::runtime_error'
    what():  [TensorRT-LLM Error][int8gemm Runner] Failed to run cutlass int8 gemm. Error: Error Internal
    ```
17. 直觉告诉我们策略选择器能运行的代码，到插件运行阶段确又失败应该是某个参数过大导致了重复运行，所以我们调整了里面和运行阶段有关的两个参数`warmup`和`runs`，经过多次编译，运行单元测试，发现3和10是比较合适的数字，可以产生较长的日志（说明能运行较长时间）。但是最终单元测试还是未能通过，后面我们还是试了一下smooth quant编译，结果发现可以。所以这个单元测试可能并不适合24G显存的A10来运行，或许是给A100跑的呢，相关变更可以通过[该commit](https://github.com/Tlntin/Qwen-7B-Chat-TensorRT-LLM/commit/0667e03b726a18a9a52e8242ddaf517f90c0e16f)查看，此时smooth_quant可以编译与运行，但是结果并不对，需要进一步校准代码。
18. 我们重新跑了gpt2的示例代码，并且发现按照readme的操作做smooth quant，我们发现，当开启gpt attention plugin比不开效果更好，这样就可以说明gpt attention plugin是支持int8 smooth quant的。我们的qwen比和gpt用了同一个attention plugin，所以如果能把rope计算放到plugin里面，那么自然SmoothAttention结构就能完全对齐了。后面我们测试了trt-llm版，fp16模式下，把外部的rope注释，将rotary_embedding_dim从0重新设置为`self.rotary_embedding_dim,`，然后我们重新build以及run然后加上summarize，完全是ok的，rouge分数和之前一样，但是summarize总计用时明显减少了3-4秒。保险起见，我们又测试新rope计算下的int8/int4 wight only，测试结果表明

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：
##### 精度
- 报告与原始模型进行精度对比测试的结果，验证精度达标（abs(rouge_diff) < 1）。
- 测试平台：NVIDIA A10 (24G显存) | TensorRT 9.0.0.1
- 最大输入长度：2048， 最大新增长度：2048，beam=batch=1
- 测试结果（该结果由`tensorrt_llm_july-release-v1/examples/qwen/summarize.py`生成）：
```bash
HuggingFace (dtype: bf16 | total latency: 99.70530200004578 sec)
  rouge1 : 28.219357100978343
  rouge2 : 9.369007098940832
  rougeL : 19.198723845033232
  rougeLsum : 22.37342869203733

TensorRT-LLM (dtype: fp16 | total latency: 70.68618655204773 sec)
  rouge1 : 28.24200534394352
  rouge2 : 9.385498589891833
  rougeL : 19.22414575248309
  rougeLsum : 22.408209721264484

TensorRT-LLM (dtype: int8 (weight only) | total latency: 45.96261024475098 sec)
  rouge1 : 29.40729169624385
  rouge2 : 10.226448500880512
  rougeL : 19.8951718521485
  rougeLsum : 23.218727152508027

TensorRT-LLM (dtype: int4 (weight only) | total latency: 33.62176060676575 sec)
  rouge1 : 29.737523789256326
  rouge2 : 11.12480662569741
  rougeL : 20.010690925962344
  rougeLsum : 24.04248574675328

```

##### 性能
- 例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
- 一般用原始模型作为baseline
- 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。
- 测试平台：NVIDIA A10 (24G显存) | TensorRT 9.0.0.1
- 测试结果（该结果由`tensorrt_llm_july-release-v1/examples/qwen/benchmark.py`生成）
1. 最大输入长度：2048， 最大新增长度：2048，num-prompts=100, beam=1, seed=0

| 测试平台     | 加速方式                  | max_batch_size | 吞吐速度（requests/s） | 生成速度（tokens/s） | 吞吐加速比 | 生成加速比 |
| ------------ | ------------------------- | -------------- | ---------------------- | -------------------- | ---------- | ---------- |
| HuggingFace  | dtype: bf16               | 1              | 0.12                   | 60.45                | 1          | 1          |
| HuggingFace  | dtype: bf16               | 2              | OOM                    | OOM                  | /          | /          |
|              |                           |                |                        |                      |            |            |
| TensorRT-LLM | dtype: fp16               | 1              | 0.18                   | 88.73                | 1.50       | 1.46       |
| TensorRT-LLM | dtype: fp16               | 2              | 0.22                   | 115.23               | 1.83       | 1.90       |
| TensorRT-LLM | dtype: fp16               | 3              | OOM                    | OOM                  | /          | /          |
|              |                           |                |                        |                      |            |            |
| TensorRT-LLM | dtype: int8 (weight only) | 1              | 0.30                   | 147.38               | 2.50       | 2.44       |
| TensorRT-LLM | dtype: int8 (weight only) | 2              | 0.31                   | 162.60               | 2.58       | 2.69       |
| TensorRT-LLM | dtype: int8 (weight only) | 3              | 0.34                   | 185.65               | 2.83       | 3.07       |
| TensorRT-LLM | dtype: int8 (weight only) | 4              | 0.36                   | 198.46               | 3.00       | 3.28       |
| TensorRT-LLM | dtype: int8 (weight only) | 5              | OOM                    | OOM                  | /          | /          |
|              |                           |                |                        |                      |            |            |
| TensorRT-LLM | dtype: int4 (weight only) | 1              | 0.43                   | 210.88               | 3.58       | 3.49       |
| TensorRT-LLM | dtype: int4 (weight only) | 2              | 0.44                   | 223.55               | 3.67       | 3.70       |
| TensorRT-LLM | dtype: int4 (weight only) | 3              | 0.46                   | 247.34               | 3.83       | 4.09       |
| TensorRT-LLM | dtype: int4 (weight only) | 4              | **0.47**               | 252.80               | **3.92**   | 4.18       |
| TensorRT-LLM | dtype: int4 (weight only) | 5              | 0.45                   | 246.13               | 3.75       | 4.07       |
| TensorRT-LLM | dtype: int4 (weight only) | 6              | **0.47**               | **261.53**           | **3.92**   | **4.33**   |
| TensorRT-LLM | dtype: int4 (weight only) | 7              | OOM                    | OOM                  | /          | /          |

2. 最大输入长度：1024， 最大新增长度：1024，num-prompts=100, beam=1, seed=0

| 测试平台     | 加速方式                  | max_batch_size | 吞吐量（requests/s） | 生成速度（tokens/s） | 吞吐加速比 | 生成加速比 |
| ------------ | ------------------------- | -------------- | -------------------- | -------------------- | ---------- | ---------- |
| HuggingFace  | dtype: bf16               | 1              | 0.14                 | 51.48                | 1          | 1          |
| HuggingFace  | dtype: bf16               | 2              | 0.12                 | 53.87                | 0.86       | 1.05       |
| HuggingFace  | dtype: bf16               | 3              | OOM                  | OOM                  | /          | /          |
|              |                           |                |                      |                      |            |            |
| TensorRT-LLM | dtype: fp16               | 1              | 0.20                 | 74.75                | 1.43       | 1.45       |
| TensorRT-LLM | dtype: fp16               | 2              | 0.23                 | 91.80                | 1.64       | 1.70       |
| TensorRT-LLM | dtype: fp16               | 3              | 0.26                 | 108.50               | 1.86       | 2.01       |
| TensorRT-LLM | dtype: fp16               | 4              | 0.29                 | 123.58               | 2.07       | 2.29       |
| TensorRT-LLM | dtype: fp16               | 5              | 0.31                 | 136.06               | 2.21       | 2.53       |
| TensorRT-LLM | dtype: fp16               | 6              | 0.31                 | 137.69               | 2.21       | 2.56       |
| TensorRT-LLM | dtype: fp16               | 7              | OOM                  | OOM                  | /          | /          |
|              |                           |                |                      |                      |            |            |
| TensorRT-LLM | dtype: int8 (weight only) | 1              | 0.34                 | 128.52               | 2.43       | 2.39       |
| TensorRT-LLM | dtype: int8 (weight only) | 2              | 0.34                 | 139.42               | 2.43       | 2.59       |
| TensorRT-LLM | dtype: int8 (weight only) | 3              | 0.37                 | 158.25               | 2.64       | 2.94       |
| TensorRT-LLM | dtype: int8 (weight only) | 4              | 0.40                 | 175.28               | 2.86       | 3.25       |
| TensorRT-LLM | dtype: int8 (weight only) | 5              | 0.44                 | 193.93               | 3.14       | 3.60       |
| TensorRT-LLM | dtype: int8 (weight only) | 6              | 0.41                 | 184.75               | 2.93       | 3.43       |
| TensorRT-LLM | dtype: int8 (weight only) | 7              | 0.46                 | 206.81               | 3.29       | 3.84       |
| TensorRT-LLM | dtype: int8 (weight only) | 8              | 0.43                 | 195.05               | 3.07       | 3.62       |
| TensorRT-LLM | dtype: int8 (weight only) | 9              | 0.47                 | 208.96               | 3.36       | 4.06       |
| TensorRT-LLM | dtype: int8 (weight only) | 10             | 0.47                 | 214.72               | 3.36       | 4.17       |
| TensorRT-LLM | dtype: int8 (weight only) | 11             | 0.45                 | 205.00               | 3.21       | 3.98       |
| TensorRT-LLM | dtype: int8 (weight only) | 12             | OOM                  | OOM                  | /          | /          |
|              |                           |                |                      |                      |            |            |
| TensorRT-LLM | dtype: int4 (weight only) | 1              | 0.53                 | 193.57               | 3.79       | 3.76       |
| TensorRT-LLM | dtype: int4 (weight only) | 2              | 0.49                 | 197.93               | 3.50       | 3.84       |
| TensorRT-LLM | dtype: int4 (weight only) | 3              | 0.52                 | 220.01               | 3.71       | 4.27       |
| TensorRT-LLM | dtype: int4 (weight only) | 4              | 0.52                 | 224.38               | 3.71       | 4.36       |
| TensorRT-LLM | dtype: int4 (weight only) | 5              | 0.58                 | 253.80               | 4.14       | 4.93       |
| TensorRT-LLM | dtype: int4 (weight only) | 6              | 0.56                 | 248.43               | 4.00       | 4.83       |
| TensorRT-LLM | dtype: int4 (weight only) | 7              | 0.56                 | 248.66               | 4.00       | 4.83       |
| TensorRT-LLM | dtype: int4 (weight only) | 8              | 0.58                 | 258.64               | 4.14       | 5.02       |
| TensorRT-LLM | dtype: int4 (weight only) | 9              | 0.58                 | 259.04               | 4.14       | 5.03       |
| TensorRT-LLM | dtype: int4 (weight only) | 10             | **0.60**             | **269.60**           | **4.29**   | **5.24**   |
| TensorRT-LLM | dtype: int4 (weight only) | 11             | 0.54                 | 246.04               | 3.86       | 4.78       |
| TensorRT-LLM | dtype: int4 (weight only) | 12             | 0.52                 | 236.83               | 3.71       | 4.60       |
| TensorRT-LLM | dtype: int4 (weight only) | 13             | 0.54                 | 246.67               | 3.86       | 4.79       |
| TensorRT-LLM | dtype: int4 (weight only) | 14             | OOM                  | OOM                  | /          | /          |

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

- 提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

 - 目前已提交的针对TensorRT的bug链接（已由导师复核确定）：[Bug1](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/86), [Bug2](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/89)

 - 已提交，待复核的bug：[Bug3](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/90)

### 送分题答案 | [操作步骤](SEND_POINT_README.md)
1. 第一题。
- 题目内容：
```text
请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）模型为gpt2-medium
python3 run.py --max_output_len=8
```
- 输出结果
```bash
Input: Born in north-east France, Soyer trained as a
Output:  chef and eventually became a chef at a
```

2. 第二题
- 题目内容
```text
请在报告中写出 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数（10分）模型为gpt2-medium
python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf --batch_size1 --test_trt_llm --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14
```

- 输出结果
```bash
TensorRT-LLM (total latency: 3.0498504638671875 sec)
TensorRT-LLM beam 0 result
  rouge1 : 21.869322054781037
  rouge2 : 6.258925475911645
  rougeL : 16.755771650012953
  rougeLsum : 18.68034777724496
Hugging Face (total latency: 9.381023168563843 sec)
HF beam 0 result
  rouge1 : 22.08914935260929
  rouge2 : 6.127009262128831
  rougeL : 16.982143879321
  rougeLsum : 19.04670077160925
```

