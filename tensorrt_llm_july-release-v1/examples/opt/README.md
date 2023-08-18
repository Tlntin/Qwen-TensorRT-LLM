# OPT

This document explains how to build the [OPT](https://huggingface.co/docs/transformers/model_doc/opt) model using TensorRT-LLM and run on a single GPU, a single node with
multiple GPUs or multiple nodes with multiple GPUs.

## Overview

The TensorRT-LLM OPT implementation can be found in [`tensorrt_llm/models/opt/model.py`](../../tensorrt_llm/models/opt/model.py). The TensorRT-LLM OPT example
code is located in [`examples/opt`](./). There are four main files in that folder:

 * [`hf_opt_convert.py`](./hf_opt_convert.py) to convert a checkpoint from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
    format to the [FasterTransformer (FT)](https://github.com/NVIDIA/FasterTransformer) format,
 * [`build.py`](./build.py) to build the [TensorRT](https://developer.nvidia.com/tensorrt) engine(s) needed to run the OPT model,
 * [`summarize.py`](./summarize.py) to summarize the articles in the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset using the model.

## Usage

The next two sections describe how to convert the weights from the [HuggingFace (HF) Transformers](https://github.com/huggingface/transformers)
format to the FT format. You can skip those two sections if you already have weights in the
FT format.

Note, also, that if your weights are neither in HF Transformers nor in FT formats, you will need to convert to the FT format. The script like
[`hf_opt_convert.py`](./hf_opt_convert.py) can serve as a starting point.

### 1. Download weights from HuggingFace Transformers

You have to make sure `git-lfs` is properly installed to load the checkpoints.

```bash
pip install -r requirements.txt && sudo apt-get install git-lfs
```

There are four different checkpoints available. Use one of the following commands to fetch the checkpoint you are interested in.

```bash
# OPT-125M
git-lfs clone https://huggingface.co/facebook/opt-125m

# OPT-350M
git-lfs clone https://huggingface.co/facebook/opt-350m

# OPT-2.7B
git-lfs clone https://huggingface.co/facebook/opt-2.7b

# OPT-66B
git-lfs clone https://huggingface.co/facebook/opt-66b
```

### 2. Convert weights from HF Tranformers to FT format

TensorRT-LLM can directly load weights from FT. The [`hf_opt_convert.py`](./hf_opt_convert.py) script allows you to convert weights from HF Tranformers
format to FT format.

```bash
# OPT-125M
python3 hf_opt_convert.py -i opt-125m -o ./c-model/opt-125m/fp16 -i_g 1 -weight_data_type fp16

# OPT-350M
python3 hf_opt_convert.py -i opt-350m -o ./c-model/opt-350m/fp16 -i_g 1 -weight_data_type fp16

# OPT-2.7B
python3 hf_opt_convert.py -i opt-2.7b -o ./c-model/opt-2.7b/fp16 -i_g 1 -weight_data_type fp16

# OPT-66B
python3 hf_opt_convert.py -i opt-66b  -o ./c-model/opt-66b/fp16  -i_g 4 -weight_data_type fp16
```

### 3. Build TensorRT engine(s)

TensorRT-LLM builds TensorRT engine(s) using a checkpoint in FT format. If no checkpoint directory is specified, TensorRT-LLM will build engine(s) using
dummy weights. Note that the number of TensorRT engines depends on the number of GPUs that will be used to run inference.

The [`build.py`](./build.py) script requires a single GPU to build the TensorRT engine(s). However, if you have more than one GPU in your system (of the same
model), you can enable parallel builds to accelerate the engine building process. For that, add the `--parallel_build` argument to the build command. We use that option for the 66B model that we split across four different GPUs.

Examples of build invocations:

```bash
# OPT-125M
python3 build.py --model_dir=./c-model/opt-125m/fp16/1-gpu \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --world_size 1 \
                 --output_dir trt_engine/opt-125m/fp16/1-gpu \
                 --do_layer_norm_before \
                 --pre_norm \
                 --hidden_act relu

# OPT-350M
python3 build.py --model_dir=./c-model/opt-350m/fp16/1-gpu \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --world_size 1 \
                 --output_dir trt_engine/opt-350m/fp16/1-gpu \
                 --post_norm \
                 --hidden_act relu

# OPT-2.7B
python3 build.py --model_dir=./c-model/opt-2.7b/fp16/1-gpu \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --world_size 1 \
                 --output_dir trt_engine/opt-2.7b/fp16/1-gpu \
                 --do_layer_norm_before \
                 --pre_norm \
                 --hidden_act relu

# OPT-66B
python3 build.py --model_dir=./c-model/opt-66b/fp16/4-gpu \
                 --max_batch_size 8 \
                 --dtype float16 \
                 --use_gpt_attention_plugin float16 \
                 --use_gemm_plugin float16 \
                 --use_layernorm_plugin float16 \
                 --max_input_len 924 \
                 --max_output_len 100 \
                 --world_size 4 \
                 --output_dir trt_engines/opt-66b/fp16/4-gpu \
                 --do_layer_norm_before \
                 --pre_norm \
                 --hidden_act relu \
                 --parallel_build
```

### 4. Summarization using the OPT model

The following section describes how to run a TensorRT-LLM OPT model to summarize the articles from the
[cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the
[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) scores and use the `ROUGE-1` score to validate the implementation.
The script can also perform the same summarization using the HF OPT model.

```bash
# OPT-125M
python3 summarize.py --engine_dir trt_engine/opt-125m/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location opt-125m \
                     --data_type fp16 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=14

# OPT-350M
python3 summarize.py --engine_dir trt_engine/opt-350m/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location opt-350m \
                     --data_type fp16 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=20

# OPT-2.7B
python3 summarize.py --engine_dir trt_engine/opt-2.7b/fp16/1-gpu \
                     --test_hf \
                     --batch_size 1 \
                     --test_trt_llm \
                     --hf_model_location opt-2.7b \
                     --data_type fp16 \
                     --check_accuracy \
                     --tensorrt_llm_rouge1_threshold=21

# OPT-66B
mpirun -n 4 --allow-run-as-root \
    python3 summarize.py --engine_dir trt_engines/opt-66b/fp16/4-gpu \
                         --batch_size 1 \
                         --test_trt_llm \
                         --hf_model_location opt-66b \
                         --data_type fp16 \
                         --check_accuracy \
                         --tensorrt_llm_rouge1_threshold=21
```

#### Fused MultiHead Attention (FMHA)

You can enable the FMHA kernels for OPT by adding `--enable_context_fmha` to the invocation of `build.py`. Note that it is disabled by default because of possible accuracy issues due to the use of Flash Attention.

If you find that the default fp16 accumulation (`--enable_context_fmha`) cannot meet the requirement, you can try to enable fp32 accumulation by adding `--enable_context_fmha_fp32_acc`. However, it is expected to see performance drop.

Note `--enable_context_fmha` / `--enable_context_fmha_fp32_acc` has to be used together with `--use_gpt_attention_plugin float16`.
