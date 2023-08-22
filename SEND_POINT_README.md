### 送分题操作步骤
##### 准备工作(可选，针对网络不好的用户可以这样干)
1. 准备一个文件夹models,将模型和数据都放里面
2. 从huggingface下载模型到服务器，然后将其复制到examples/gpt目录下，并且重命名为gpt2
```bash
cp -r models/gpt2-medium tensorrt_llm_july-release-v1/examples/gpt/gpt2
```

3. 从huggingface下载数据集并缓存到本地，然后将缓存复制到examples/gpt目录下。
```bash
python3 -c "from datasets import load_dataset;dataset = load_dataset('ccdv/cnn_dailymail', '3.0.0',cache_dir='models/cnn_dailymail')"
mkdir tensorrt_llm_july-release-v1/examples/gpt/ccdv
cp -r models/cnn_dailymail tensorrt_llm_july-release-v1/examples/gpt/ccdv
```

4. 进入examples/gpt目录
```bash
cd tensorrt_llm_july-release-v1/examples/gpt
```

5. 安装3个基本py模块，否则会报错。
```bash
pip install datasets
pip install nltk
pip install rouge_score
```


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
3. 执行最后一个命令, 计算pytorch版和TRT版的`rouge_score`(这一步需要能`和谐上网`，否则无法执行)
```bash
python3 summarize.py --engine_dir trt_engine/gpt2/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location=gpt2 \
                     --check_accuracy \
                     --dataset_path="ccdv/cnn_dailymail" \
                     --tensorrt_llm_rouge1_threshold=14
```
