# TensorRT-LLM: A TensorRT toolbox for Large Language Models

## Table of Contents

- [The TensorRT-LLM Overview](#the-tensorrt-llm-overview)
- [Installation](#installation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Release notes](#release-notes)
  - [Changelog](#changelog)
  - [Known issues](#known-issues)

## The TensorRT-LLM Overview

TensorRT-LLM provides users with an easy-to-use Python API to define Large
Language Models (LLMs) and build
[TensorRT](https://developer.nvidia.com/tensorrt) engines that contain
state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs.
TensorRT-LLM also contains components to create Python and C++ runtimes that
execute those TensorRT engines. It also includes a backend for integration with
the [NVIDIA Triton Inference
Server](https://developer.nvidia.com/nvidia-triton-inference-server).  Models
built with TensorRT-LLM can be executed on a wide range of configurations going
from a single GPU to multiple nodes with multiple GPUs (using Tensor
Parallelism).

The Python API of TensorRT-LLM is architectured to look similar to the
[PyTorch](https://pytorch.org) API. It provides users with a
[functional](./tensorrt_llm/functional.py) module containing functions like
`einsum`, `softmax`, `matmul` or `view`. The [layer](./tensorrt_llm/layer)
module bundles useful building blocks to assemble LLMs; like an `Attention`
block, a `MLP` or the entire `Transformer` layer. Model-specific components,
like `GPTAttention` or `BertAttention`, can be found in the
[model](./tensorrt_llm/model) module.

TensorRT-LLM provides users with predefined models that can easily be modified
and extended. The current version of TensorRT-LLM supports
[BERT](https://huggingface.co/docs/transformers/model_doc/bert),
[GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt),
[NVIDIA GPT-2B](https://huggingface.co/nvidia/GPT-2B-001),
[GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj),
[LLaMA](https://huggingface.co/docs/transformers/model_doc/llama),
[OPT](https://huggingface.co/docs/transformers/model_doc/opt),
[SantaCoder](https://huggingface.co/bigcode/santacoder)
and
[StarCoder](https://huggingface.co/bigcode/starcoder).
To maximize performance and reduce memory footprint, TensorRT-LLM allows the
models to be executed using different quantization modes (see
[`examples/gpt`](./examples/gpt) for concrete examples).  TensorRT-LLM supports
INT4 or INT8 weights (and FP16 activations; a.k.a.  INT4/INT8 weight-only) as
well as a complete implementation of the
[SmoothQuant](https://arxiv.org/abs/2211.10438) technique.

For a more detailed presentation of the software architecture and the key
concepts used in TensorRT-LLM, we recommend you to read the following
[document](./docs/architecture.md).

## Installation

### Docker Container

We recommend that you use a [Docker](https://www.docker.com) container to build
and run TensorRT-LLM. Instructions to install an environment to run Docker
containers for the NVIDIA platform can be found
[here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

To create a Docker container to build and run TensorRT-LLM, you need to
download the following packages from NVOnline:

 * `polygraphy-0.48.1-py2.py3-none-any.whl`,
 * `TensorRT-9.0.0.2.Linux.x86_64-gnu.cuda-12.2.tar.gz`.

Copy those packages to the top-level directory of TensorRT-LLM. Then, use the
following command:

```bash
DOCKER_BUILDKIT=1 docker build -t tensorrt_llm -f docker/Dockerfile.dev .
```

To run the container, use the following command:

```bash
docker run --gpus all --rm -it -v ${PWD}:/code/tensorrt_llm -w /code/tensorrt_llm tensorrt_llm bash
```

### Build from Source

Make sure you have fetched all the dependencies before compiling TensorRT-LLM:

```bash
git submodule update --init --recursive
```

Once it is done, you can build the code from inside that container using:

```bash
# To build the TensorRT-LLM code.
./scripts/build_wheel.py --trt_root /usr/local/TensorRT-9.0.0.2

# Deploy TensorRT-LLM in your environment.
pip install ./build/tensorrt_llm*.whl
```

By default, `build_wheel.py` enables incremental builds. To clean the build
directory, add the `--clean` option:

```bash
./scripts/build_wheel.py --clean --trt_root /usr/local/TensorRT-9.0.0.2

```

### Building for Specific CUDA Architectures

Specific CUDA architectures may be passed as an argument to
[`build_wheel.py`](scripts/build_wheel.py). The script accepts a single
argument taking a semicolon separated list of CUDA architecture specifications
compatible with [CUDA_ARCHITECTURES in CMake].  For instance, to build for
compute capabilities 8.0 and 8.6, call `build_wheel.py` like so:

```bash
./scripts/build_wheel.py --cuda_architectures "80-real;86-real" --trt_root /usr/local/TensorRT-9.0.0.2
```

### Building and Linking against the C++ Runtime of TensorRT-LLM

Running `build_wheel.py` will also compile the library containing the C++
runtime of TensorRT-LLM. If Python support and `torch` modules are not
required, the script provides the option `--cpp_only` which restricts the build
to the C++ runtime only:

```bash
./scripts/build_wheel.py --cuda_architectures "80-real;86-real" --cpp_only --clean --trt_root /usr/local/TensorRT-9.0.0.2
```

This is particularly useful to avoid linking problems which may be introduced
by particular versions of `torch` related to the [dual ABI support of
GCC](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). The
option `--clean` will remove the build directory before building. The default
build directory is `cpp/build`, which may be overridden using the option
`--build_dir`. Run `build_wheel.py --help` for an overview of all supported
options.

Clients may choose to link against the shared or the static version of the
library. These libraries can be found in the following locations:

```bash
cpp/build/tensorrt_llm/libtensorrt_llm.so
cpp/build/tensorrt_llm/libtensorrt_llm_static.a
```

In addition, one needs to link against the library containing the LLM plugins
for TensorRT available here:

```bash
cpp/build/tensorrt_llm/plugins/libnvinfer_plugin.so
```

Add the following directories to your project include paths

```bash
cpp
cpp/include
```

Only header files contained in `cpp/include` are part of the supported API and
may be directly included. Other headers contained under `cpp` should not be
included directly since they might change in future versions.

For examples of how to use the C++ runtime, see the unit tests in
[gptSessionTest.cpp](cpp/tests/runtime/gptSessionTest.cpp) and the related
[CMakeLists.txt](cpp/tests/CMakeLists.txt) file.

## Examples

- [Bert](examples/bert)
- [BLOOM](examples/bloom)
- [ChatGLM-6B](examples/chatglm6b)
- [GPT](examples/gpt)
- [GPT-J](examples/gptj)
- [GPT-NeoX](examples/gptneox)
- [LLaMA](examples/llama)
- [OpenAI Triton](examples/openai_triton)
- [OPT](examples/opt)

## Troubleshooting

- It's recommended to add options `–shm-size=1g –ulimit memlock=-1` to the
  docker or nvidia-docker run command.  Otherwise you may see NCCL errors when
  running multiple GPU inferences, see
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html#errors
  for details.

- If you encounter
```text
NVIDIA H100 PCIe with CUDA capability sm_90 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
```

when building engines, you need to install the preview version of PyTorch that
corresponds to your CUDA version.  As an example, for CUDA 12.1, use:

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

[CUDA_ARCHITECTURES in CMake]: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES

## Release notes

### Changelog

**July 2023**

  - TensorRT-LLM requires TensorRT 9.0,
  - Support for BLOOM, ChatGLM 6B, GPT-NeoX, LLaMA v2,
  - Support for BF16 and FP8 models,
  - Support for in-flight batching,
  - Support for a new C++ Triton Backend,
  - Refactoring of the KV cache to support pagging,
    - The KV cache is now decomposed into blocks,
    - The layout of the K cache has changed to `[batch_size, num_heads, seq_length, dim_per_head]`,
  - Support for multi-GPU embeddings,
  - Support for embedding sharing (input embedding and LM head),
  - New example that shows how to integrate an OpenAI Triton kernel into TensorRT-LLM,
  - Improved documentation (Docstrings in `functional.py` and documentation in `docs`)

**June 2023**

  - Support Nemo-GPT Next, SantaCoder, StarCoder in FP16,
  - Support for a new C++ Runtime (with streaming support),
  - Support for beam-search,
  - Support for Multiquery Attention (MQA),
  - Support for RoPE,
  - Support for INT8 KV Cache,
  - Support INT4 weight-only (with GPT example), but the weight-only kernels will not be optimal on hopper

**May 2023**

  - **The initial release of TensorRT-LLM**
  - Support GPT, BERT, OPT, LLaMA in FP16,
  - Support single-node multi-GPU GPT, OPT, BERT, LLaMA FP16 using Tensor parallelism,
  - Support Triton Inference Server with a Python backend,
  - Support sampling features, including top-k, top-p, temperature, and sampling penalty,
  - Attention support
   - Optimized Flash-Attention-based Multihead Attention for Ampere, Ada and Hopper architectures,
   - Multi-Query Attention (MQA),
   - ALiBi in Multihead-Attention,
  - Support SmoothQuant INT8 (with GPT example),
  - Support INT8 weight-only (with GPT example), but the weight-only kernels will not be optimal on hopper

### Known issues
