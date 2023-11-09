# Qwen

This document shows how to build and run a Qwen model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM Qwen implementation can be found in [model.py](model.py). The TensorRT-LLM Qwen example code is located in [`examples/qwen`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Qwen model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix
  * FP16
    <!-- * FP8 -->
  * INT8 & INT4 Weight-Only
    <!-- * FP8 KV CACHE -->
  * INT8 KV CACHE
  * Tensor Parallel
  * STRONGLY TYPED

## Usage

The TensorRT-LLM Qwen example code locates at [examples/qwen](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF Qwen checkpoint first by following the guides here [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) or [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)

Create a `tmp/Qwen` directory to store the weights downloaded from huaggingface.
```bash
mkdir -p ./tmp/Qwen
```

Store Qwen-7B-Chat or Qwen-14B-Chat separately.
- for Qwen-7B-Chat
```bash
mv Qwen-7B-Chat ./tmp/Qwen/7B
```
- for Qwen-14B-Chat
```
mv Qwen-14B-Chat ./tmp/Qwen/14B
```

TensorRT-LLM Qwen builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in Qwen.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

# Build the Qwen 7B model using a single GPU and FP16.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# Build the Qwen 7B model using a single GPU and BF16.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/

# Build the Qwen 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int8 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/

# Build the Qwen 7B model using a single GPU and apply INT4 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/

# Build Qwen 7B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen 7B using 2-way tensor parallelism and apply INT4 weight-only quantization..
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen 7B using 2-way tensor parallelism and 2-way pipeline parallelism.
python build.py --hf_model_dir ./tmp/Qwen/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

# Build Qwen 14B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen/14B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2
```


#### INT8 weight only + INT8 KV cache
For INT8 KV cache, [`hf_qwen_convert.py`](./hf_qwen_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_qwen_convert.py \
    -i ./tmp/Qwen/7B/ \
    -o ./tmp/Qwen/7B/int8_kv_cache/ \
    --calibrate-kv-cache -t float16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_dir_path ./tmp/Qwen/7B/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --hf_model_dir ./tmp/Qwen/7B \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

- run
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu
```

Test with `summarize.py`:


- validate huggingface
```bash
python3 summarize.py --backend=hf \
    --tokenizer_dir ./tmp/Qwen/7B \
    --hf_model_dir ./tmp/Qwen/7B
```

- validate trt-llm
```bash
python3 summarize.py --backend=trt_llm \
    --tokenizer_dir ./tmp/Qwen/7B \
    --engine_dir ./tmp/Qwen/7B/trt_engines/int8_kv_cache_weight_only/1-gpu 
```

#### SmoothQuant

The smoothquant supports both Qwen v1 and Qwen v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_qwen_convert.py -i ./tmp/Qwen/7B -o ./tmp/Qwen/7B/sq0.5/ -sq 0.5 --tensor-parallelism 1 --storage-type float16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --ft_dir_path=./tmp/Qwen/7B/sq0.5/1-gpu/ \
                 --use_smooth_quant \
                 --hf_model_dir ./tmp/Qwen/7B \
                 --output_dir ./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --ft_dir_path=./tmp/Qwen/7B/sq0.5/1-gpu/ \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel \
                 --hf_model_dir ./tmp/Qwen/7B \
                 --output_dir ./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/
```

- run
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/
```

- summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir=./tmp/Qwen/7B/trt_engines/sq0.5/1-gpu/
```

#### INT4-GPTQ
To run the GPTQ Qwen example, the following steps are required:
1. You need to install the [auto-gptq](https://github.com/PanQiWei/AutoGPTQ) module and upgrade the transformers module version, with a minimum of 4.32.0. (Note: After installing the module, it may prompt that the tensorrt_llm is not compatible with other module versions, you can ignore this warning)
```bash
pip install auto-gptq
pip install transformers -U
```

2. Weight quantization
```bash
python3 gptq_cpu_convert.py --hf_model_dir ./tmp/Qwen/7B \
							--tokenizer_dir ./tmp/Qwen/7B \
                 			--quant_ckpt_path ./tmp/Qwen/7B/int4-gptq
```

3. Build TRT-LLM engine:
```bash
python build.py --hf_model_dir ./tmp/Qwen/7B \
                --quant_ckpt_path ./tmp/Qwen/7B/int4-gptq/gptq_model-4bit-128g.safetensors \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_gptq \
                --per_group \
                --world_size 1 \
                --tp_size 1 \
                --output_dir ./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu
```

4. Run int4-gptq
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu
```

5. Summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen/7B/trt_engines/int4-gptq/1-gpu
```

### Run

To run a TensorRT-LLM Qwen model using the engines generated by build.py

```bash
# With fp16 inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# With fp16 inference two gpus
mpirun -n 2 --allow-run-as-root  \
    python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/fp16/2-gpu/

# With bf16 inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/bf16/1-gpu

# With int8 weight only inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/

# With int4 weight only inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/

# With int4 weight only inference use two gpus
mpirun -n 2 --allow-run-as-root  \
    python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen/7B/ \
               --engine_dir=./tmp/Qwen/7B/trt_engines/int4_weight_only/2-gpu/
```

### Summarization using the Qwen model

```bash
# Run summarization using the Qwen 7B model in FP16.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/1-gpu/

# Run summarization using the Qwen 7B model in BF16.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen/7B/trt_engines/bf16/1-gpu/

# Run summarization using the Qwen 7B model quantized to INT8.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir  ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen/7B/trt_engines/int8_weight_only/1-gpu/

# Run summarization using the Qwen 7B model quantized to INT4.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir  ./tmp/Qwen/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen/7B/trt_engines/int4_weight_only/1-gpu/

# Run summarization using the Qwen 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --backend=trt_llm \
                        --tokenizer_dir  ./tmp/Qwen/7B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/Qwen/7B/trt_engines/fp16/2-gpu/

# Run summarization using the Qwen 14B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --backend=trt_llm \
                        --tokenizer_dir  ./tmp/Qwen/14B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/Qwen/14B/trt_engines/fp16/2-gpu/
```
