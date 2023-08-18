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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class TllmRuntime;
class GptDecoderBatch;

class GptSessionWithDecoderBatch
{
public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    GptSessionWithDecoderBatch(GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr);

    GptSessionWithDecoderBatch(GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : GptSessionWithDecoderBatch(modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), logger)
    {
    }

    GptSessionWithDecoderBatch(GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::string const& engineFile, LoggerPtr logger = nullptr)
        : GptSessionWithDecoderBatch(modelConfig, worldConfig, loadEngine(engineFile), logger)
    {
    }

    [[nodiscard]] GptModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] WorldConfig const& getWorldConfig() const
    {
        return mWorldConfig;
    }

    [[nodiscard]] int getDevice() const noexcept
    {
        return mDevice;
    }

    void setup(SizeType batchSize, SizeType maxInputLength, SizeType maxNewTokens, SizeType endId, SizeType padId,
        SamplingConfig const& samplingConfig);

    void generate(
        GenerationOutput& outputs, GenerationInput const& inputs, std::vector<SamplingConfig> const& samplingConfig);

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager& getBufferManager() const;

    static std::vector<uint8_t> loadEngine(std::string const& enginePath);

private:
    using TensorMap = StringPtrMap<ITensor>;
    using TensorPtr = ITensor::SharedPtr;

    void createContexts();
    void createDecoder(SizeType endId, SizeType padId);

    void getContextBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, TensorPtr const& inputIds,
        TensorPtr const& inputLengths) const;
    void getNextStepBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType step, TensorPtr const& inputIds,
        TensorPtr const& inputLengths) const;

    void prepareContextStep(GenerationInput const& inputs);
    TensorPtr prepareNextStep(SizeType step, bool inputPacked, TensorPtr const& outputIds);

    void copyInputToOutput(TensorPtr const& outputIds, TensorPtr const& inputIds, TensorPtr const& inputLengths,
        SizeType padId, SizeType maxInputLength, bool inputPacked) const;

    void gatherTree(
        TensorPtr& outputIds, TensorPtr const& sequenceLength, TensorPtr const& endIds, SizeType finalSeqLength) const;

    void checkInputShape(TensorPtr const& inputIds, TensorPtr const& inputLengths, bool inputPacked) const;

    static int initDevice(WorldConfig const& worldConfig);

    static void setRawPointers(
        TensorPtr& pointers, TensorPtr const& input, int32_t leadingDimMultiplier, const nvinfer1::DataType& type);

    template <typename T>
    static void setRawPointers_(TensorPtr& pointers, TensorPtr const& input, int32_t leadingDimMultiplier)
    {
        const auto shape = input->getShape();
        auto requestShape = shape;
        requestShape.d[0] = 1;
        const auto stride = ITensor::volume(requestShape) * leadingDimMultiplier;

        const auto basePtr = bufferCast<T>(*input);

        auto pointersPtr = bufferCast<SizeType>(*pointers);
        for (SizeType reqIdx = 0; reqIdx < shape.d[0] / leadingDimMultiplier; ++reqIdx)
        {
            const auto ptr = reinterpret_cast<std::uintptr_t>(basePtr + stride * reqIdx);
            pointersPtr[2 * reqIdx + 0] = ptr & 0xffffffff;
            pointersPtr[2 * reqIdx + 1] = (ptr >> 32) & 0xffffffff;
        }
    }

    class GenerationConfig
    {
    public:
        GenerationConfig() = default;

        GenerationConfig(SizeType batchSize, SizeType maxInputLength, SizeType maxNewTokens, SizeType beamWidth)
            : batchSize{batchSize}
            , maxInputLength{maxInputLength}
            , maxNewTokens{maxNewTokens}
            , beamWidth{beamWidth}
        {
        }

        SizeType batchSize{};
        SizeType maxInputLength{};
        SizeType maxNewTokens{};
        SizeType beamWidth{};

        SizeType getMaxSeqLength() const
        {
            return maxInputLength + maxNewTokens;
        }
    };

    class RuntimeBuffers
    {
    public:
        // general
        TensorPtr inputLengths;
        TensorPtr inputOffsets;

        // engine
        TensorPtr logits;
        TensorPtr sequenceLengths;     // with attention plugin
        TensorPtr pastKeyValueLengths; // with attention plugin
        TensorPtr tokenMask;           // with attention plugin
        TensorPtr attentionMask;       // without attention plugin
        TensorPtr positionIds;
        TensorPtr lastTokenIds;
        TensorPtr maxInputLength;

        // Inputs specific to Inflight Batching Attention plugin
        TensorPtr beamWidths;
        TensorPtr inputLengths2;
        std::vector<TensorPtr> keyValuePointers;
        TensorPtr requestCacheMaxSeqLengths;
        TensorPtr hostInputLengths;
        TensorPtr cacheIndirectionPointers;

        std::vector<TensorPtr> presentKeysVals;
        std::vector<TensorPtr> presentKeysValsAlt;

        // decoder
        TensorPtr cumLogProbs;
        TensorPtr parentIds;

        // beam search (shared between engine and decoder)
        TensorPtr cacheIndirection0;
        TensorPtr cacheIndirection1;

        bool allocated{false};

        void clear();

        void create(TllmRuntime& runtime, GptModelConfig const& modelConfig);

        void reshape(GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, SizeType worldSize);

        void tile(BufferManager& manager, GenerationConfig const& generationConfig, bool usePlugin);
    };

    GptModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};

private:
    GenerationConfig mGenerationConfig{};

    LoggerPtr mLogger;
    std::shared_ptr<TllmRuntime> mRuntime;
    std::shared_ptr<GptDecoderBatch> mDecoder;

    RuntimeBuffers mBuffers;
};

} // namespace tensorrt_llm::runtime
