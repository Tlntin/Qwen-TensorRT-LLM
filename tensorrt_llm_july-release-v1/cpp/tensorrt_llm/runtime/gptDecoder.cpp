/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/gptDecoder.h"

#include "tensorrt_llm/common/tensorConversion.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"

#include <memory>

#include <NvInferRuntime.h>

namespace tc = tensorrt_llm::common;
namespace tl = tensorrt_llm::layers;
namespace tcc = tensorrt_llm::common::conversion;

using namespace tensorrt_llm::runtime;

template <typename T>
GptDecoder<T>::GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream)
    : mManager{stream}
    , mAllocator{mManager}
{
    tc::cublasMMWrapper* cublasWrapper = nullptr;
    bool isFreeBufferAfterForward{false};
    cudaDeviceProp prop;
    tc::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    mDynamicDecodeLayer = std::make_shared<tensorrt_llm::layers::DynamicDecodeLayer<T>>(
        vocabSize, vocabSizePadded, stream->get(), cublasWrapper, &mAllocator, isFreeBufferAfterForward, &prop);
}

template <typename T>
void GptDecoder<T>::setup(SamplingConfig const& samplingConfig, size_t batchSize)
{
    typename layers::DynamicDecodeLayer<T>::SetupParams setupParams;

    setupParams.random_seed = samplingConfig.randomSeed;

    setupParams.repetition_penalty = samplingConfig.repetitionPenalty;
    setupParams.presence_penalty = samplingConfig.presencePenalty;
    setupParams.temperature = samplingConfig.temperature;
    setupParams.min_length = samplingConfig.minLength;

    // signed to unsigned
    if (samplingConfig.topK)
    {
        auto const& topK = samplingConfig.topK.value();
        setupParams.runtime_top_k = std::vector<uint32_t>(std::begin(topK), std::end(topK));
    }

    setupParams.runtime_top_p = samplingConfig.topP;
    setupParams.top_p_decay = samplingConfig.topPDecay;
    setupParams.top_p_min = samplingConfig.topPMin;
    setupParams.top_p_reset_ids = samplingConfig.topPResetIds;

    setupParams.beam_search_diversity_rate = samplingConfig.beamSearchDiversityRate;
    setupParams.length_penalty = samplingConfig.lengthPenalty;

    mDynamicDecodeLayer->setup(batchSize, samplingConfig.beamWidth, setupParams);
}

namespace
{
void safeInsert(tc::TensorMap& map, std::string const& key, DecodingOutput::TensorPtr const& tensor)
{
    if (tensor)
    {
        ITensor const& t{*tensor};
        map.insert({key, tcc::toTllmTensor(t)});
    }
}

template <typename T>
typename tl::DynamicDecodeLayer<T>::ForwardParams prepareInputs(DecodingInput const& input)
{
    auto constexpr ite = 0; // no pipeline parallelism
    typename tl::DynamicDecodeLayer<T>::ForwardParams forwardParams{input.step, ite, input.maxLength, input.batchSize,
        tcc::toTllmTensor(*input.logits), tcc::toTllmTensor(*input.endIds)};

    if (input.cacheIndirection)
    {
        forwardParams.src_cache_indirection = tcc::toTllmTensor(*input.cacheIndirection);
    }

    if (input.sequenceLimitLength)
    {
        forwardParams.sequence_limit_length = tcc::toTllmTensor(*input.sequenceLimitLength);
    }

    if (input.embeddingBias)
    {
        forwardParams.embedding_bias = tcc::toTllmTensor(*input.embeddingBias);
    }

    if (input.lengths)
    {
        forwardParams.input_lengths = tcc::toTllmTensor(*input.lengths);
    }

    if (input.badWordsList)
    {
        forwardParams.bad_words_list = tcc::toTllmTensor(*input.badWordsList);
    }

    if (input.stopWordsList)
    {
        forwardParams.stop_words_list = tcc::toTllmTensor(*input.stopWordsList);
    }

    return forwardParams;
}

template <typename T>
typename tl::DynamicDecodeLayer<T>::OutputParams prepareOutputs(DecodingOutput& output)
{
    typename tl::DynamicDecodeLayer<T>::OutputParams outputParams(tcc::toTllmTensor(*output.ids));

    if (output.cumLogProbs)
    {
        outputParams.cum_log_probs = tcc::toTllmTensor(*output.cumLogProbs);
    }

    if (output.parentIds)
    {
        outputParams.parent_ids = tcc::toTllmTensor(*output.parentIds);
    }

    if (output.cacheIndirection)
    {
        outputParams.tgt_cache_indirection = tcc::toTllmTensor(*output.cacheIndirection);
    }

    if (output.finished)
    {
        outputParams.finished = tcc::toTllmTensor(*output.finished);
    }

    if (output.finishedSum)
    {
        outputParams.finished_sum = tcc::toTllmTensor(*output.finishedSum);
    }

    if (output.lengths)
    {
        outputParams.sequence_length = tcc::toTllmTensor(*output.lengths);
    }

    if (output.logProbs)
    {
        outputParams.output_log_probs = tcc::toTllmTensor(*output.logProbs);
    }

    return outputParams;
}

} // namespace

template <typename T>
bool GptDecoder<T>::forward(DecodingOutput& output, DecodingInput const& input)
{
    auto forwardParams = prepareInputs<T>(input);
    auto outputParams = prepareOutputs<T>(output);

    BufferManager::ITensorPtr finishedSum;
    std::int32_t* finishedSumHost = nullptr;
    if (input.sequenceLimitLength && output.finished)
    {
        if (output.finishedSum)
        {
            finishedSumHost = bufferCast<std::int32_t>(*output.finishedSum);
        }
        else
        {
            finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
            outputParams.finished_sum = tcc::toTllmTensor(*finishedSum);
            finishedSumHost = bufferCast<std::int32_t>(*finishedSum);
        }
        *finishedSumHost = 0;
    }

    mDynamicDecodeLayer->forward(outputParams, forwardParams);

    if (finishedSumHost)
    {
        auto const numToFinish = output.finished->getSize();
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(mDynamicDecodeLayer->getStream()));
        return numToFinish == static_cast<std::size_t>(*finishedSumHost);
    }
    else
    {
        return false;
    }
}

template <typename T>
void GptDecoder<T>::forwardAsync(DecodingOutput& output, DecodingInput const& input)
{
    auto forwardParams = prepareInputs<T>(input);
    auto outputParams = prepareOutputs<T>(output);

    mDynamicDecodeLayer->forward(outputParams, forwardParams);
}

namespace tensorrt_llm::runtime
{
template class GptDecoder<float>;
template class GptDecoder<half>;
} // namespace tensorrt_llm::runtime
