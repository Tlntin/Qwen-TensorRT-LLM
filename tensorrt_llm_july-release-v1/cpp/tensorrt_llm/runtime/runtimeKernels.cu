/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::kernels
{

namespace
{

template <typename T>
__global__ void fill(T* data, std::size_t size, T const value)
{
    auto const idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        data[idx] = value;
    }
}
} // namespace

template <typename T>
void invokeFill(IBuffer& buffer, T const value, CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const size = buffer.getSize();
    dim3 const blockSize(256);
    dim3 const gridSize((size + blockSize.x - 1) / blockSize.x);

    fill<<<gridSize, blockSize, 0, stream.get()>>>(data, size, value);
}

// template instantiation
template void invokeFill(IBuffer&, SizeType, CudaStream const&);
template void invokeFill(IBuffer&, float, CudaStream const&);

namespace
{
template <typename T>
__global__ void add(T* data, std::size_t size, T const value)
{
    auto const idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        data[idx] += value;
    }
}
} // namespace

template <typename T>
void invokeAdd(IBuffer& buffer, T const value, CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const size = buffer.getSize();
    dim3 const blockSize(256);
    dim3 const gridSize((size + blockSize.x - 1) / blockSize.x);

    add<<<gridSize, blockSize, 0, stream.get()>>>(data, size, value);
}

template void invokeAdd(IBuffer&, SizeType, CudaStream const&);

namespace
{
__global__ void set2(SizeType* data, SizeType const value0, SizeType const value1)
{
    SizeType const idx = static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < 2)
    {
        data[idx] = (idx == 0) ? value0 : value1;
    }
}
} // namespace

template <typename T>
void invokeSet2(IBuffer& buffer, T const value0, T const value1, CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const size = buffer.getSize();
    TLLM_CHECK_WITH_INFO(size == 2, "buffer has wrong size");
    dim3 const blockSize(32);
    dim3 const gridSize(1);

    set2<<<gridSize, blockSize, 0, stream.get()>>>(data, value0, value1);
}

template void invokeSet2(IBuffer&, SizeType, SizeType, CudaStream const&);

namespace
{
__global__ void transpose(SizeType* output, SizeType const* input, SizeType const batchSize, SizeType const rowSize)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < rowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * rowSize + tokenIdx;
            auto const outputIdx = tokenIdx * batchSize + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(output.getDataType() == input.getDataType(), "input and output have different data types");
    TLLM_CHECK_WITH_INFO(output.getSize() == input.getSize(), "input and output have different sizes");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(inputShape.nbDims == 2, "input shape must have 2 dimensions");

    SizeType const batchSize = inputShape.d[0];
    SizeType const rowSize = inputShape.d[1];

    dim3 const blockSize(256, 1);
    dim3 const gridSize((rowSize + blockSize.x - 1) / blockSize.x, batchSize);

    transpose<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<SizeType>(output), bufferCast<SizeType const>(input), batchSize, rowSize);
}

namespace
{
__global__ void transposeWithOutputOffset(SizeType* output, SizeType const* input, SizeType const nbInputRows,
    SizeType const inputRowSize, SizeType const outputRowSize, SizeType const outputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < nbInputRows; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < inputRowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + tokenIdx;
            auto const outputIdx = tokenIdx * outputRowSize + outputOffset + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType const outputOffset, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(output.getDataType() == input.getDataType(), "input and output have different data types");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(inputShape.nbDims == 2, "input shape must have 2 dimensions");
    SizeType const nbInputRows = inputShape.d[0];
    SizeType const inputRowSize = inputShape.d[1];

    auto const& outputShape = output.getShape();
    TLLM_CHECK_WITH_INFO(outputShape.nbDims == 2, "output shape must have 2 dimensions");
    SizeType const nbOutputRows = outputShape.d[0];
    SizeType const outputRowSize = outputShape.d[1];

    TLLM_CHECK_WITH_INFO(nbOutputRows == inputRowSize, "input dim 1 and output dim 0 have different sizes");
    TLLM_CHECK_WITH_INFO(outputOffset + nbInputRows <= outputRowSize, "input does not fit into output tensor");

    dim3 const blockSize(256, 1);
    dim3 const gridSize((inputRowSize + blockSize.x - 1) / blockSize.x, nbInputRows);

    transposeWithOutputOffset<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(output),
        bufferCast<SizeType const>(input), nbInputRows, inputRowSize, outputRowSize, outputOffset);
}

namespace
{
__global__ void transposeWithInputOffset(SizeType* output, SizeType const* input, SizeType const outputRowSize,
    SizeType const nbOutputRows, SizeType const inputRowSize, SizeType const inputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < outputRowSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < nbOutputRows; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + inputOffset + tokenIdx;
            auto const outputIdx = tokenIdx * outputRowSize + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType const inputOffset, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(output.getDataType() == input.getDataType(), "input and output have different data types");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(inputShape.nbDims == 2, "input shape must have 2 dimensions");
    SizeType const nbInputRows = inputShape.d[0];
    SizeType const inputRowSize = inputShape.d[1];

    auto const& outputShape = output.getShape();
    SizeType const nbOutputRows = outputShape.d[0];
    SizeType const outputRowSize = outputShape.d[1];

    TLLM_CHECK_WITH_INFO(outputShape.nbDims == 2, "output shape must have 2 dimensions");
    TLLM_CHECK_WITH_INFO(outputRowSize == nbInputRows, "input dim 0 and output dim 1 have different sizes");
    TLLM_CHECK_WITH_INFO(inputOffset + nbOutputRows <= inputRowSize, "input does not fit into output tensor");

    dim3 const blockSize(256, 1);
    dim3 const gridSize((nbOutputRows + blockSize.x - 1) / blockSize.x, outputRowSize);

    transposeWithInputOffset<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(output),
        bufferCast<SizeType const>(input), outputRowSize, nbOutputRows, inputRowSize, inputOffset);
}

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream)
{
    auto const size = input.getSize();
    auto const* inputData = bufferCast<SizeType>(input);
    auto* outputData = bufferCast<SizeType>(output);

    std::size_t tempStorageBytes{0};
    cub::DeviceScan::InclusiveSum(nullptr, tempStorageBytes, inputData, outputData, size, stream.get());
    auto tempStorage = manager.gpu(tempStorageBytes, nvinfer1::DataType::kUINT8);
    auto* tempStorageData = bufferCast<std::uint8_t>(*tempStorage);
    cub::DeviceScan::InclusiveSum(tempStorageData, tempStorageBytes, inputData, outputData, size, stream.get());
}

namespace
{
__global__ void buildTokenMask(SizeType* tokenMask, SizeType const* inputLengths, SizeType const batchSize,
    SizeType const maxInputLength, SizeType const maxSeqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType tokenIdx = tidx; tokenIdx < maxSeqLength; tokenIdx += blockDim.x * gridDim.x)
        {
            tokenMask[batchIdx * maxSeqLength + tokenIdx]
                = (tokenIdx >= inputLength && tokenIdx < maxInputLength) ? 1 : 0;
        }
    }
}
} // namespace

void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType const maxInputLength, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == tokenMask.getDataType(), "tokenMask has wrong data type");
    TLLM_CHECK_WITH_INFO(
        TRTDataType<SizeType>::value == inputLengths.getDataType(), "inputLengths has wrong data type");

    auto const& shape = tokenMask.getShape();
    SizeType const batchSize = shape.d[0];
    SizeType const maxSeqLength = shape.d[1];

    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength, "tokenMask dimension 2 is too small");

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxSeqLength + blockSize.x - 1) / blockSize.x, batchSize);

    buildTokenMask<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(tokenMask),
        bufferCast<SizeType const>(inputLengths), batchSize, maxInputLength, maxSeqLength);
}

namespace
{
__global__ void buildAttentionMask(SizeType* attentionMask, SizeType const size, SizeType const padId)
{
    SizeType const tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (SizeType i = tid; i < size; i += blockDim.x * gridDim.x)
    {
        auto const x = attentionMask[i];
        attentionMask[i] = (x != padId);
    }
}
} // namespace

void invokeBuildAttentionMask(ITensor& attentionMask, SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        TRTDataType<SizeType>::value == attentionMask.getDataType(), "attentionMask has wrong data type");

    auto const size = attentionMask.getSize();
    dim3 const blockSize(256);
    dim3 const gridSize((size + blockSize.x - 1) / blockSize.x);

    buildAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(attentionMask), size, padId);
}

namespace
{
__global__ void extendAttentionMask(
    SizeType* newMask, SizeType const* oldMask, SizeType const batchSize, SizeType const seqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < seqLength; tokenIdx += blockDim.x * gridDim.x)
        {
            SizeType oldIndex = batchIdx * seqLength + tokenIdx;
            SizeType newIndex = batchIdx * (seqLength + 1) + tokenIdx;
            newMask[newIndex] = (tokenIdx < seqLength) ? oldMask[oldIndex] : 1;
        }
    }
}
} // namespace

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == newMask.getDataType(), "attentionMask has wrong data type");
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == oldMask.getDataType(), "attentionMask has wrong data type");

    auto const& shape = oldMask.getShape();
    SizeType const batchSize = shape.d[0];
    SizeType const seqLength = shape.d[1];

    dim3 const blockSize(256, 1);
    dim3 const gridSize((seqLength + blockSize.x - 1) / blockSize.x, batchSize);

    extendAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<SizeType>(newMask), bufferCast<SizeType>(oldMask), batchSize, seqLength);
}

namespace
{
__global__ void copyInputToOutput(SizeType* outputIds, SizeType const* inputIds, SizeType const* inputLengths,
    SizeType const padId, SizeType const batchSize, SizeType const beamWidth, SizeType const maxInputLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[batchIdx * maxInputLength + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(tokenIdx, batchIdx, beamIdx, batchSize, beamWidth);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        outputIds.getDataType() == inputIds.getDataType(), "input and output ids have different data types");

    auto const batchSize = static_cast<SizeType>(inputLengths.getSize());
    auto const& inputShape = inputIds.getShape();
    SizeType const maxInputLength = inputShape.d[inputShape.nbDims - 1];
    auto const& outputShape = outputIds.getShape();
    SizeType const maxSeqLength = outputShape.d[0];
    SizeType const beamWidth = outputShape.d[2];

    TLLM_CHECK_WITH_INFO(
        std::size_t(batchSize) == inputIds.getSize() / maxInputLength, "input ids have wrong batch size");
    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[1], "output ids have wrong batch size");
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength, "output ids have to be larger than input ids");

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputLengths), padId, batchSize, beamWidth,
        maxInputLength);
}

namespace
{
__global__ void copyPackedInputToOutput(SizeType* outputIds, SizeType const* inputIds, SizeType const* inputOffsets,
    SizeType const padId, SizeType const batchSize, SizeType const beamWidth, SizeType const maxInputLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const tokenBegin = inputOffsets[batchIdx];
        auto const tokenEnd = inputOffsets[batchIdx + 1];
        auto const inputLength = tokenEnd - tokenBegin;

        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[tokenBegin + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(tokenIdx, batchIdx, beamIdx, batchSize, beamWidth);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType const maxInputLength, SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        outputIds.getDataType() == inputIds.getDataType(), "input and output ids have different data types");

    auto const batchSize = static_cast<SizeType>(inputOffsets.getSize()) - 1;
    auto const& outputShape = outputIds.getShape();
    SizeType const maxSeqLength = outputShape.d[0];
    SizeType const beamWidth = outputShape.d[2];

    TLLM_CHECK_WITH_INFO(batchSize == (outputShape.d[1]), "output ids have wrong batch size");
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength, "output ids have to be larger than input ids");

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyPackedInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputOffsets), padId, batchSize, beamWidth,
        maxInputLength);
}

namespace
{
template <typename T>
__global__ void tileTensor(T* output, T const* input, SizeType const batchSize, SizeType const inputRowSize,
    SizeType const outputRowSize, SizeType const beamWidth)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType columnIdx = tidx; columnIdx < inputRowSize; columnIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + columnIdx;
            auto const value = input[inputIdx];
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = (batchIdx * beamWidth + beamIdx) * outputRowSize + columnIdx;
                output[outputIdx] = value;
            }
        }
    }
}
} // namespace

template <typename T>
void invokeTileTensor(ITensor& output, ITensor const& input, SizeType const beamWidth, CudaStream const& stream)
{
    auto const& inputShape = input.getShape();
    auto const nbInputRows = inputShape.d[0];
    auto const inputRowSize = static_cast<SizeType>(input.getSize()) / nbInputRows;
    auto const& outputShape = output.getShape();
    auto const nbOutputRows = outputShape.d[0];
    auto const outputRowSize = static_cast<SizeType>(output.getSize()) / nbOutputRows;

    TLLM_CHECK_WITH_INFO(nbOutputRows == beamWidth * nbInputRows,
        common::fmtstr(
            "nbOutputRows (%d) must be beamWidth (%d) times nbInputRows (%d)", nbOutputRows, beamWidth, nbInputRows));
    TLLM_CHECK_WITH_INFO(outputRowSize >= inputRowSize,
        common::fmtstr("output row size (%d) must be at least input row size (%d)", outputRowSize, inputRowSize));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((inputRowSize + blockSize.x - 1) / blockSize.x, nbInputRows);
    tileTensor<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<T>(output), bufferCast<T const>(input), nbInputRows, inputRowSize, outputRowSize, beamWidth);
}

template void invokeTileTensor<SizeType>(
    ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);
template void invokeTileTensor<float>(
    ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);
template void invokeTileTensor<half>(
    ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);
template void invokeTileTensor<int8_t>(
    ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream);

} // namespace tensorrt_llm::runtime::kernels
