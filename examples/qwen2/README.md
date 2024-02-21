# Qwen1.5(Qwen2-beta)

This document shows how to build and run a Qwen1.5 model in TensorRT-LLM on both single GPU, single node multi-GPU and multi-node multi-GPU.

## Overview

The TensorRT-LLM Qwen1.5 implementation can be found in [model.py](model.py). The TensorRT-LLM Qwen1.5 example code is located in [`examples/Qwen1.5`](./). There are three main files in that folder::

 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the Qwen1.5 model,
 * [`run.py`](./run.py) to run the inference on an input text,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Support Matrix
  * FP16
    <!-- * FP8 -->
  * INT8 & INT4 Weight-Only & INT4-AWQ & INT4-GPTQ
    <!-- * FP8 KV CACHE -->
  * INT8 KV CACHE
  * Tensor Parallel
  * STRONGLY TYPED

## Support Model
- Qwen1.5 1.8b/4b/7b/14b/72b(maybe)
- Qwen1.5 1.8b-chat/4b-chat/7b-chat/14b-chat/72b-chat(maybe)
- Qwen1.5 1.8b-chat-int4-gptq/4b-chat-int4-gptq/7b-chat-int4-gptq/14b-chat-int4-gptq/72b-chat-int4-gptq(maybe)

## Usage

The TensorRT-LLM Qwen1.5 example code locates at [examples/Qwen2](./). It takes HF weights as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build TensorRT engine(s)

Need to prepare the HF Qwen1.5 checkpoint first by following the guides here [Qwen1.5-7B-Chat](https://huggingface.co/Qwen1.5/Qwen1.5-7B-Chat) or [Qwen1.5-14B-Chat](https://huggingface.co/Qwen1.5/Qwen1.5-14B-Chat)

Create a `tmp/Qwen1.5` directory to store the weights downloaded from huaggingface.
```bash
mkdir -p ./tmp/Qwen1.5
```

Store Qwen1.5-7B-Chat or Qwen1.5-14B-Chat separately.
- for Qwen1.5-7B-Chat
```bash
mv Qwen1.5-7B-Chat ./tmp/Qwen1.5/7B
```
- for Qwen1.5-14B-Chat
```
mv Qwen1.5-14B-Chat ./tmp/Qwen1.5/14B
```

TensorRT-LLM Qwen1.5 builds TensorRT engine(s) from HF checkpoint. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) with dummy weights.

Normally `build.py` only requires single GPU, but if you've already got all the GPUs needed while inferencing, you could enable parallelly building to make the engine building process faster by adding `--parallel_build` argument. Please note that currently `parallel_build` feature only supports single node.

Here're some examples:

```bash
# Build a single-GPU float16 engine from HF weights.
# use_gpt_attention_plugin is necessary in Qwen1.5.
# Try use_gemm_plugin to prevent accuracy issue.
# It is recommend to use --remove_input_padding along with --use_gpt_attention_plugin for better performance

# Build the Qwen1.5 7B model using a single GPU and FP16.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/fp16/1-gpu/

# Build the Qwen1.5 7B model using a single GPU and BF16.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype bfloat16 \
                --remove_input_padding \
                --use_gpt_attention_plugin bfloat16 \
                --enable_context_fmha \
                --use_gemm_plugin bfloat16 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/bf16/1-gpu/

# Build the Qwen1.5 7B model using a single GPU and apply INT8 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int8 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int8_weight_only/1-gpu/

# Build the Qwen1.5 7B model using a single GPU and apply INT4 weight-only quantization.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int4_weight_only/1-gpu/

# Build Qwen1.5 7B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen1.5 7B using 2-way tensor parallelism and apply INT4 weight-only quantization..
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int4_weight_only/2-gpu/ \
                --world_size 2 \
                --tp_size 2

# Build Qwen1.5 7B using 2-way tensor parallelism and 2-way pipeline parallelism.
python build.py --hf_model_dir ./tmp/Qwen1.5/7B/ \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/fp16/2-gpu/ \
                --world_size 4 \
                --tp_size 2 \
                --pp_size 2

# Build Qwen1.5 14B using 2-way tensor parallelism.
python build.py --hf_model_dir ./tmp/Qwen1.5/14B \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen1.5/14B/trt_engines/fp16/2-gpu/ \
                --world_size 2 \
                --tp_size 2
```


#### INT8 weight only + INT8 KV cache
For INT8 KV cache, [`hf_Qwen1.5_convert.py`](./hf_Qwen1.5_convert.py) features a
`--calibrate-kv-cache, -kv` option. Setting `-kv` will calibrate the model,
and then export the scaling factors needed for INT8 KV cache inference.


Example:

```bash
python3 hf_Qwen1.5_convert.py \
    -i ./tmp/Qwen1.5/7B/ \
    -o ./tmp/Qwen1.5/7B/int8_kv_cache/ \
    --calibrate-kv-cache -t float16
```

[`build.py`](./build.py) add new options for the support of INT8 KV cache.

`--int8_kv_cache` is the command-line option to enable INT8 KV cache.

In addition, it could be combined with INT8 weight-only quantization, as follows:

Examples of INT8 weight-only quantization + INT8 KV cache

```bash
# Build model with both INT8 weight-only and INT8 KV cache enabled
python build.py --ft_dir_path ./tmp/Qwen1.5/7B/int8_kv_cache/1-gpu/ \
                --dtype float16 \
                --hf_model_dir ./tmp/Qwen1.5/7B \
                --use_gpt_attention_plugin float16 \
                --use_gemm_plugin float16 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int8_kv_cache_weight_only/1-gpu \
                --int8_kv_cache \
                --use_weight_only
```

- run
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int8_kv_cache_weight_only/1-gpu
```

Test with `summarize.py`:


- validate huggingface
```bash
python3 summarize.py --backend=hf \
    --tokenizer_dir ./tmp/Qwen1.5/7B \
    --hf_model_dir ./tmp/Qwen1.5/7B
```

- validate trt-llm
```bash
python3 summarize.py --backend=trt_llm \
    --tokenizer_dir ./tmp/Qwen1.5/7B \
    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/int8_kv_cache_weight_only/1-gpu 
```

#### SmoothQuant

The smoothquant supports both Qwen1.5 v1 and Qwen1.5 v2. Unlike the FP16 build where the HF weights are processed and loaded into the TensorRT-LLM directly, the SmoothQuant needs to load INT8 weights which should be pre-processed before building an engine.

Example:
```bash
python3 hf_Qwen1.5_convert.py -i ./tmp/Qwen1.5/7B -o ./tmp/Qwen1.5/7B/sq0.5/ -sq 0.5 --tensor-parallelism 1 --storage-type float16
```

[`build.py`](./build.py) add new options for the support of INT8 inference of SmoothQuant models.

`--use_smooth_quant` is the starting point of INT8 inference. By default, it
will run the model in the _per-tensor_ mode.

Then, you can add any combination of `--per-token` and `--per-channel` to get the corresponding behaviors.

Examples of build invocations:

```bash
# Build model for SmoothQuant in the _per_tensor_ mode.
python3 build.py --ft_dir_path=./tmp/Qwen1.5/7B/sq0.5/1-gpu/ \
                 --use_smooth_quant \
                 --hf_model_dir ./tmp/Qwen1.5/7B \
                 --output_dir ./tmp/Qwen1.5/7B/trt_engines/sq0.5/1-gpu/

# Build model for SmoothQuant in the _per_token_ + _per_channel_ mode
python3 build.py --ft_dir_path=./tmp/Qwen1.5/7B/sq0.5/1-gpu/ \
                 --use_smooth_quant \
                 --per_token \
                 --per_channel \
                 --hf_model_dir ./tmp/Qwen1.5/7B \
                 --output_dir ./tmp/Qwen1.5/7B/trt_engines/sq0.5/1-gpu/
```

- run
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/sq0.5/1-gpu/
```

- summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir=./tmp/Qwen1.5/7B/trt_engines/sq0.5/1-gpu/
```

#### INT4-GPTQ
To run the GPTQ Qwen1.5 example, the following steps are required:
1. You need to install the [auto-gptq](https://github.com/PanQiWei/AutoGPTQ) module and upgrade the transformers module version, with a minimum of 4.32.0. (Note: After installing the module, it may prompt that the tensorrt_llm is not compatible with other module versions, you can ignore this warning)
```bash
pip install auto-gptq optimum
pip install transformers -U
```
2. Manually get the quanted weights (optional)
- Weight quantization
```bash
python3 gptq_convert.py --hf_model_dir ./tmp/Qwen1.5/7B \
                        --tokenizer_dir ./tmp/Qwen1.5/7B \
                        --quant_ckpt_path ./tmp/Qwen1.5/7B/int4-gptq
```

-  Build TRT-LLM engine:
```bash
python build.py --hf_model_dir ./tmp/Qwen1.5/7B \
                --quant_ckpt_path ./tmp/Qwen1.5/7B/int4-gptq/gptq_model-4bit-128g.safetensors \
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
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int4-gptq/1-gpu
```

- Run int4-gptq
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int4-gptq/1-gpu
```

- Summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/int4-gptq/1-gpu
```

3. Use official int4 weights, e.g. Qwen1.5-1_8B-Chat-Int4 model(recommended)
-  Build TRT-LLM engine:
```bash
python build.py --hf_model_dir Qwen1.5-1_8B-Chat-Int4 \
                --quant_ckpt_path Qwen1.5-1_8B-Chat-Int4 \
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
                --output_dir ./tmp/Qwen1.5/1.8B/trt_engines/int4-gptq/1-gpu
```

- Run int4-gptq
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir Qwen1.5-1_8B-Chat-Int4 \
               --engine_dir=./tmp/Qwen1.5/1.8B/trt_engines/int4-gptq/1-gpu
```

- Summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir Qwen1.5-1_8B-Chat-Int4 \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/1.8B/trt_engines/int4-gptq/1-gpu
```


#### INT4-AWQ
To run the AWQ Qwen1.5 example, the following steps are required:
1. Download and install the [nvidia-ammo](https://developer.nvidia.com/downloads/assets/cuda/files/nvidia-ammo/nvidia_ammo-0.3.0.tar.gz) module. An installation code is required below. For reference, be careful not to install the cuda version, but the universal version, otherwise there will be bugs.
```bash
pip install nvidia_ammo-0.3.0-cp310-cp310-linux_x86_64.whl
```
2. Modify the ammo code and add Qwen1.5 support (an error will be reported if not added). Here is a simple reference case:
- First, write a python file in vscode and import the following function
```python
from tensorrt_llm.models.quantized.ammo import quantize_and_export
```
- Then control + left mouse button, click the `quantize_and_export` function to view its internal implementation.
- Find "model_lookup" , add the following question code to support Qwen1.5
```bash
    ("Qwen2", ): "Qwen2",
```
- After modification, it looks like this:
```bash
model_lookup = {
    ("llama", "mistral"): "llama",
    ("gptj", ): "gptj",
    ("falcon", "rw"): "falcon",
    ("baichuan", ): "baichuan",
    ("mpt", ): "mpt",
    ("gpt2", ): "gpt2",
    ("chatglm", ): "chatglm",
    ("Qwen2", ): "Qwen2",
}
```
- Second, change save code
- before
```python
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
```python
if export_path:
    with torch.inference_mode():
        if qformat == "int4_awq" and model_type == "Qwen1.5":
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
3. Weight quantization
```bash
python3 quantize.py --model_dir ./tmp/Qwen1.5/7B \
                    --dtype float16 \
                    --qformat int4_awq \
                    --export_path ./Qwen1.5_7b_4bit_gs128_awq.pt \
                    --calib_size 32
```

4. TRT-LLM engine:
```bash
python build.py --hf_model_dir ./tmp/Qwen1.5/7B \
                --quant_ckpt_path ./Qwen1.5_7b_4bit_gs128_awq.pt \
                --dtype float16 \
                --remove_input_padding \
                --use_gpt_attention_plugin float16 \
                --enable_context_fmha \
                --use_gemm_plugin float16 \
                --use_weight_only \
                --weight_only_precision int4_awq \
                --per_group \
                --world_size 1 \
                --tp_size 1 \
                --output_dir ./tmp/Qwen1.5/7B/trt_engines/int4-awq/1-gpu
```
5. Run int4-gptq
```bash
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int4-awq/1-gpu
```

6. Summarize
```bash
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/int4-awq/1-gpu


### Run

To run a TensorRT-LLM Qwen1.5 model using the engines generated by build.py

```bash
# With fp16 inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/fp16/1-gpu/

# With fp16 inference two gpus
mpirun -n 2 --allow-run-as-root  \
    python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/fp16/2-gpu/

# With bf16 inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/bf16/1-gpu

# With int8 weight only inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int8_weight_only/1-gpu/

# With int4 weight only inference
python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int4_weight_only/1-gpu/

# With int4 weight only inference use two gpus
mpirun -n 2 --allow-run-as-root  \
    python3 run.py --max_new_tokens=50 \
               --tokenizer_dir ./tmp/Qwen1.5/7B/ \
               --engine_dir=./tmp/Qwen1.5/7B/trt_engines/int4_weight_only/2-gpu/
```

### Summarization using the Qwen1.5 model

```bash
# Run summarization using the Qwen1.5 7B model in FP16.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/fp16/1-gpu/

# Run summarization using the Qwen1.5 7B model in BF16.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/bf16/1-gpu/

# Run summarization using the Qwen1.5 7B model quantized to INT8.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir  ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/int8_weight_only/1-gpu/

# Run summarization using the Qwen1.5 7B model quantized to INT4.
python summarize.py --backend=trt_llm \
                    --tokenizer_dir  ./tmp/Qwen1.5/7B/ \
                    --data_type fp16 \
                    --engine_dir ./tmp/Qwen1.5/7B/trt_engines/int4_weight_only/1-gpu/

# Run summarization using the Qwen1.5 7B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --backend=trt_llm \
                        --tokenizer_dir  ./tmp/Qwen1.5/7B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/Qwen1.5/7B/trt_engines/fp16/2-gpu/

# Run summarization using the Qwen1.5 14B model in FP16 using two GPUs.
mpirun -n 2 --allow-run-as-root \
    python summarize.py --backend=trt_llm \
                        --tokenizer_dir  ./tmp/Qwen1.5/14B/ \
                        --data_type fp16 \
                        --engine_dir ./tmp/Qwen1.5/14B/trt_engines/fp16/2-gpu/
```
