### 总述

- 介绍本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，本项目将使用TRT-LLM完成对Qwen-7B-Chat实现推理加速。

- 原始模型：Qwen-7B-Chat
- 原始模型URL：[Qwen-7B-Chat 🤗](https://huggingface.co/Qwen/Qwen-7B-Chat) [Qwen-7B-Chat Github](https://github.com/QwenLM/Qwen-7B)
- 选题类型：2+4（注：2指的是TRT-LLM实现新模型。4指的是在新模型上启用了TRT-LLM现有feature）

### 主要贡献

请简练地概括项目的主要贡献，使读者可以快速理解并复现你的工作，包括：

- 优化效果（例如给出精度和加速比），简单给出关键的数字即可，在这里不必详细展开
- 在Docker里面代码编译、运行步骤的完整说明
  - 请做到只要逐行运行你给的命令，就能把代码跑起来

### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。

1. huggingface转llm-trt比较繁琐。
- 目前只能肉眼观察已有成功案例，例如参考chatglm/llama, 通过debug huggingface版和观察trt-llm版，了解整体思路。
- 然后观察qwen和chatglm/llama的差异，拿这两个案例做魔改。整体代码魔改自llama, attention/rope参考了chatglm-6b。
2. 首次运行报显存分配错误。
- 在其附近插入可用显存和需要分配的显存代码，发现是显存不够, 将max_batch_size从默认的8改成2后解决。
3. 没有现成的attention实现。
- 通过对比代码，发现examples下面的chatglm-6b的rope embedding和qwen类似，所以chatglm-6b的rope embedding的trt实现可以作为参考项。
- 移植时发现,rope提前算好了weights，然后分割成了两个cos_embedding和sin_embedding。为确保该方案可行，于是在huggingface版的qwen中实现了类似结构，对比rope_cos和rope_sim的输出结果，以及对应sum值，发现该操作可行，于是将其移植到了qwen trt-llm中。
- 不过需要注意的是，qwen的dim和max_position_dim和chatglm-6b不一样，加上chatglm-6b trt-llm的rope的inv_freq做了一定约分，导致看起来比较奇怪，所以后续我们直接使用了的qwen原版的inv_freq计算，以及qwen原版的apply_rotary_pos_emb方法。
4. fp16下，模型的logits无法对齐。
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

5. trt-llm输出结果和pytorch不一致。
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
- 后续我们将原版的后处理函数`_decode_chatml`应用到run.py后，多余字符被完全清除（为了和原版的model.chat对齐，<|im_end|>等特殊字符也被清除）。
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
您好，我是来自达摩院的大规模语言模型，我叫通义千问。
"""
```

- 至此，在trt-llm上支持qwen模型的基础工作已经做完

- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

- 精度：报告与原始模型进行精度对比测试的结果，验证精度达标。
  - 如果选用TensorRT-LLM，请跑summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。如果精度良好，原始模型与优化模型的Rouge score的差异一般在1以内。例子见 TensorRT-LLM docker 中 /root/workspace/tensorrt_llm_july-release-v1/examples/gpt/summarize.py
  - 如果选用TensorRT，这里的精度测试指的是针对“原始模型”和“TensorRT优化模型”分别输出的数据（tensor）进行数值比较。请给出绝对误差和相对误差的统计结果（至少包括最大值、平均值与中位数）。
    - 使用训练好的权重和有意义的输入数据更有说服力。如果选手使用了随机权重和输入数据，请在这里注明。
    - 在精度损失较大的情况下，鼓励选手用训练好的权重和测试数据集对模型优化前与优化后的准确度指标做全面比较，以增强说服力。
- 性能：例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
  - 一般用原始模型作为baseline
  - 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。

请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

提交的针对TensorRT的bug链接：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/86

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

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

