### 送分题操作步骤
##### 准备工作
1. 进入examples/gpt目录
```bash
cd /app/tensorrt_llm/examples/gpt
```

2. 安装3个基本py模块，否则会报错。
```bash
pip install datasets nltk rouge_score
```
3. 从huggingface下载模型到服务器，然后将其移动到examples/gpt目录下，并且重命名为gpt2
```bash
git lfs install
git clone https://huggingface.co/gpt2-medium
mv gpt2-medium /app/tensorrt_llm/examples/gpt/gpt2
```

4. 针对`网络不好`的用户，可以通过百度网盘下载对应数据集，然后根据里面的使用说明将其解压到huggingface的cache路径。
- 百度网盘链接:https://pan.baidu.com/s/1aJrE3c6aMi7Qsc5zXk_amw?pwd=apfd 提取码:apfd


##### 送分题1执行步骤
1. 转HuggingFace模型到FT格式
```bash
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16
```

2. 将FT格式的模型数据编译成TensorRT Engine
```bash
python3 build.py --model_dir=./c-model/gpt2/1-gpu --use_gpt_attention_plugin
```

3. 跑一下推理，看看输出结果
```bash
python3 run.py --max_output_len=8
```


##### 送分题2执行步骤
1. 转HuggingFace模型到FT格式
```bash
python3 hf_gpt_convert.py -i gpt2 -o ./c-model/gpt2/fp16 --tensor-parallelism 1 --storage-type float16
```

2. 将FT格式的模型数据编译成TensorRT Engine
```bash
python3 build.py --model_dir=./c-model/gpt2/fp16/1-gpu \
                 --use_gpt_attention_plugin \
                 --use_gemm_plugin \
                 --use_layernorm_plugin \
                 --max_batch_size 8 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --output_dir trt_engine/gpt2/fp16/1-gpu/ \
                 --hidden_act gelu
```
3. 执行最后一个命令, 计算pytorch版和TRT版的`rouge_score`
```bash
python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=gpt2 \
                     --check_accuracy
```
