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
1. 完整支持原版的logn和ntk（这俩参数是用于增强模型长输出效果，这里的长输出指的是大于2048小于8192）。
2. 支持`RotaryEmbedding`，并且在input_len > 2048时开启ntk相关计算。
3. 支持`gpt_attention_plugin`与`gemm_plugin`两个plugin。
4. 同时支持qwen base和chat模型
5. 支持fp16 / int8 (weight only) / int4 (weight only), 理论上最低只需要8G消费级显卡就能运行。
6. 支持在终端对话和使用gradio构建的网页应用中对话，支持流式输出。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：
1. 精度
- 报告与原始模型进行精度对比测试的结果，验证精度达标（abs(rouge_diff) < 1）。
- 注：`datasets.load_metric("rouge")`已提示废弃，将由`evaluate.load("rouge")`代替
- 测试平台：NVIDIA A10 | TensorRT 9.0.0.1
- 测试结果（该结果由`tensorrt_llm_july-release-v1/examples/qwen/summarize.py`生成）：
```bash
Hugging Face (dtype: bf16 | total latency: 134.0561056137085 sec)
  rouge1 : 26.98400945199415
  rouge2 : 8.362191635355105
  rougeL : 18.64579951191403
  rougeLsum : 20.76437573207235

TensorRT-LLM (dtype: fp16 | total latency: 68.62463283538818 sec)
  rouge1 : 26.98400945199415
  rouge2 : 8.362191635355105
  rougeL : 18.64579951191403
  rougeLsum : 20.76437573207235

TensorRT-LLM (dtype: int8 (weight only) | total latency: 42.23632740974426 sec)
  rouge1 : 26.98263929036846
  rouge2 : 8.327280257593927
  rougeL : 18.630452012787206
  rougeLsum : 20.853083825182235

TensorRT-LLM (dtype: int4 (weight only) | total latency: 31.1434268951416 sec)
rouge1 : 26.65239213023417
rouge2 : 8.148988533684609
rougeL : 18.180307238649856
rougeLsum : 21.995243873709555

```

2. 性能（待写）
- 例如可以用图表展示不同batch size或sequence length下性能加速效果（考虑到可能模型可能比较大，可以只给batch size为1的数据）
- 一般用原始模型作为baseline
- 一般提供模型推理时间的加速比即可；若能提供压力测试下的吞吐提升则更好。


请注意：

- 相关测试代码也需要包含在代码仓库中，可被复现。
- 请写明云主机的软件硬件环境，方便他人参考。

### Bug报告（可选）

- 提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。

 - 目前已提交的针对TensorRT的bug链接（已由导师复核确定）：https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/86

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

