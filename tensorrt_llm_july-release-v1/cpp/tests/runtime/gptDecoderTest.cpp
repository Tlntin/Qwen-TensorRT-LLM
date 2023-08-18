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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/worldConfig.h"

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

namespace
{

void testDecoder(nvinfer1::DataType const dtype, SamplingConfig const& samplingConfig)
{
    SizeType constexpr worldSize{1};
    SizeType constexpr localRank{0};
    WorldConfig constexpr worldConfig{worldSize, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    bool constexpr useGptAttentionPlugin{false};
    GptModelConfig const modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype, useGptAttentionPlugin};

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    // create decoder
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = IGptDecoder::create(modelConfig.getDataType(), vocabSize, vocabSizePadded, streamPtr);
    ASSERT_TRUE(static_cast<bool>(decoder));

    // setup decoder
    auto const beamWidth = samplingConfig.beamWidth;
    SizeType constexpr batchSize{4};

    decoder->setup(samplingConfig, batchSize);

    int constexpr endId{50257};
    SizeType constexpr step{0};
    SizeType constexpr maxInputLength{8};
    auto constexpr decodeStep = maxInputLength + step;
    SizeType constexpr maxNewTokens{2};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;

    // set up inputs
    auto logits = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), dtype));
    manager.setZero(*logits);

    std::vector<int> const endIdsVec(batchSize * beamWidth, endId);
    auto endIds
        = std::shared_ptr(manager.copyFrom(endIdsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU));

    DecodingInput inputs{decodeStep, maxInputLength, batchSize, logits, endIds};
    std::vector<std::int32_t> sequenceLimitLengthsVec(batchSize, maxSeqLength - 1);
    inputs.sequenceLimitLength
        = manager.copyFrom(sequenceLimitLengthsVec, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    if (beamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), nvinfer1::DataType::kINT32));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    auto outputIds = std::shared_ptr(
        manager.gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32));
    manager.setZero(*outputIds);
    DecodingOutput outputs{outputIds};

    std::vector<int> sequenceLengthsVec(batchSize * beamWidth, maxInputLength);
    outputs.lengths
        = manager.copyFrom(sequenceLengthsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU);
    outputs.finished = manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kBOOL);
    manager.setZero(*outputs.finished);
    outputs.finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto* finishedSumHost = bufferCast<std::int32_t>(*outputs.finishedSum);
    *finishedSumHost = -1;

    if (beamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;

        auto cumLogProbs
            = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kFLOAT));
        manager.setZero(*cumLogProbs);
        outputs.cumLogProbs = cumLogProbs;

        auto parentIds = std::shared_ptr(
            manager.gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32));
        manager.setZero(*parentIds);
        outputs.parentIds = parentIds;
    }

    // run decoder
    EXPECT_FALSE(decoder->forward(outputs, inputs));
    EXPECT_EQ(*finishedSumHost, 0);

    // verify results
    auto outputsIdsHost = manager.copyFrom(*outputs.ids, MemoryType::kCPU);
    auto output = bufferCast<std::int32_t>(*outputsIdsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            auto const result = (beamWidth == 1) ? 1023 : bw;

            bool anyMismatch = false;
            for (auto i = 0; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, bw, batchSize, beamWidth);
                EXPECT_EQ(output[outputIndex], 0) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != 0);
            }
            for (auto i = 0; i < maxNewTokens - 1; ++i)
            {
                auto const index = tc::flat_index3(maxInputLength + i, b, bw, batchSize, beamWidth);
                EXPECT_EQ(output[index], result) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[index] != result);
            }
            ASSERT_FALSE(anyMismatch);
        }
    }

    // run decoder again
    inputs.step += 1;
    EXPECT_TRUE(decoder->forward(outputs, inputs));
    EXPECT_EQ(*finishedSumHost, outputs.finished->getSize());
}

} // namespace

TEST(GptDecoderTest, FloatBeamWidth1)
{
    SizeType constexpr beamWidth{1};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(nvinfer1::DataType::kFLOAT, samplingConfig);
}

TEST(GptDecoderTest, FloatBeamWidth3)
{
    SizeType constexpr beamWidth{3};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(nvinfer1::DataType::kFLOAT, samplingConfig);
}

TEST(GptDecoderTest, HalfBeamWidth1)
{
    SizeType constexpr beamWidth{1};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(nvinfer1::DataType::kHALF, samplingConfig);
}

TEST(GptDecoderTest, HalfBeamWidth3)
{
    SizeType constexpr beamWidth{3};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(nvinfer1::DataType::kHALF, samplingConfig);
}
