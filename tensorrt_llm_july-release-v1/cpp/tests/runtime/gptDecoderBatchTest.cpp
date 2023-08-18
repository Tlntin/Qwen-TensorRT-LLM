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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
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
    int constexpr endId{50257};
    int constexpr padId{50257};

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, endId, padId, streamPtr);

    // setup decoder
    auto const beamWidth = samplingConfig.beamWidth;
    SizeType constexpr batchSize{4};
    SizeType constexpr maxInputLength{8};
    SizeType constexpr maxNewTokens{2};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;

    decoder.setup(batchSize, beamWidth, maxInputLength + maxNewTokens);

    std::vector<SizeType> const inputLengths{4, 5, 6, 7};

    // set up inputs
    auto logits = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), dtype));
    manager.setZero(*logits);

    decoder_batch::Input inputs{logits};
    if (beamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), TRTDataType<SizeType>::value));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    decoder_batch::Output outputs{};

    if (beamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), TRTDataType<SizeType>::value));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;
    }

    auto constexpr tokenId = 1;
    std::vector<decoder_batch::Input::TensorPtr> inputIds;
    for (auto b = 0; b < batchSize; ++b)
    {
        auto shape = ITensor::makeShape({inputLengths[b]});
        auto input = std::shared_ptr(manager.gpu(shape, TRTDataType<SizeType>::value));
        kernels::invokeFill(*input, tokenId, *streamPtr);
        inputIds.emplace_back(input);
        decoder.newRequest(b, decoder_batch::Request{inputIds[b], maxNewTokens}, samplingConfig);
    }

    auto outputsIds = decoder.getOutputIds();
    ASSERT_TRUE(outputsIds);
    auto outputShape = outputsIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 3);
    EXPECT_EQ(outputShape.d[0], batchSize);
    EXPECT_EQ(outputShape.d[1], beamWidth);
    EXPECT_EQ(outputShape.d[2], maxSeqLength);

    auto outputsIdsHost = manager.copyFrom(*outputsIds, MemoryType::kCPU);
    auto output = bufferCast<TokenIdType>(*outputsIdsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            bool anyMismatch = false;
            for (auto i = 0; i < inputLengths[b]; ++i)
            {
                auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                EXPECT_EQ(output[outputIndex], tokenId) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != tokenId);
            }
            for (auto i = inputLengths[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                EXPECT_EQ(output[outputIndex], padId) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != padId);
            }
            ASSERT_FALSE(anyMismatch);
        }
    }

    auto& nbSteps = decoder.getNbSteps();
    EXPECT_THAT(nbSteps, ::testing::Each(0));

    // run decoder
    decoder.forward(outputs, inputs);

    // verify results
    auto& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));
    EXPECT_THAT(nbSteps, ::testing::Each(1));

    auto sequenceLengths = decoder.getOutputLengths();
    ASSERT_TRUE(sequenceLengths);
    EXPECT_EQ(sequenceLengths->getSize(), batchSize * beamWidth);
    auto sequenceLengthsHost = manager.copyFrom(*sequenceLengths, MemoryType::kCPU);
    auto sequenceLengthsPtr = bufferCast<SizeType>(*sequenceLengthsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            auto index = tc::flat_index(sequenceLengths->getShape().d, b, bw);
            EXPECT_EQ(sequenceLengthsPtr[index], inputLengths[b] + 1);
        }
    }

    outputsIds = decoder.getOutputIds();
    // TODO(nkorobov): test parentIds
    // parentIds = decoder.getParentIds();
    ASSERT_TRUE(outputsIds);
    outputShape = outputsIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 3);
    EXPECT_EQ(outputShape.d[0], batchSize);
    EXPECT_EQ(outputShape.d[1], beamWidth);
    EXPECT_EQ(outputShape.d[2], maxSeqLength);

    outputsIdsHost = manager.copyFrom(*outputsIds, MemoryType::kCPU);
    output = bufferCast<TokenIdType>(*outputsIdsHost);
    manager.getStream().synchronize();

    std::vector<std::int32_t> expected{1023, 26623, 1023, 25860};

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            auto const result
                = (beamWidth == 1) ? (dtype == nvinfer1::DataType::kFLOAT ? 1023 : expected[b % expected.size()]) : bw;

            bool anyMismatch = false;
            for (auto i = 0; i < inputLengths[b]; ++i)
            {
                auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                EXPECT_EQ(output[outputIndex], tokenId) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != tokenId);
            }
            for (auto i = inputLengths[b]; i < inputLengths[b] + maxNewTokens - 1; ++i)
            {
                auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                EXPECT_EQ(output[outputIndex], result) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != result);
            }
            for (auto i = inputLengths[b] + maxNewTokens; i < maxSeqLength; ++i)
            {
                auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                EXPECT_EQ(output[outputIndex], padId) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != padId);
            }
            if (anyMismatch)
            {
                std::cerr << "expected {";
                for (auto i = 0; i < maxSeqLength; ++i)
                {
                    if (i < inputLengths[b])
                        std::cerr << tokenId << ", ";
                    else if (i < inputLengths[b] + maxNewTokens)
                        std::cerr << result << ", ";
                    else
                        std::cerr << padId << ", ";
                }
                std::cerr << "}\n";

                std::cerr << "but got  {";
                for (auto i = 0; i < maxSeqLength; ++i)
                {
                    auto const outputIndex = tc::flat_index(outputShape.d, b, bw, i);
                    std::cerr << output[outputIndex] << ", ";
                }
                std::cerr << "}\n";
            }
            ASSERT_FALSE(anyMismatch);
        }
    }

    decoder.forward(outputs, inputs);
    EXPECT_THAT(finished, ::testing::Each(true));
    EXPECT_THAT(nbSteps, ::testing::Each(maxNewTokens));

    EXPECT_NO_THROW(decoder.forward(outputs, inputs));
    EXPECT_THAT(nbSteps, ::testing::Each(maxNewTokens));

    decoder.newRequest(0, decoder_batch::Request{inputIds[0], maxNewTokens}, samplingConfig);
    EXPECT_FALSE(finished[0]);
    EXPECT_EQ(nbSteps[0], 0);
}

} // namespace

#define TRY_LOG_EXCEPTION(val)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        try                                                                                                            \
        {                                                                                                              \
            val;                                                                                                       \
        }                                                                                                              \
        catch (std::exception const& e)                                                                                \
        {                                                                                                              \
            TLLM_LOG_EXCEPTION(e);                                                                                     \
            throw e;                                                                                                   \
        }                                                                                                              \
    } while (0)

TEST(GptDecoderBatchTest, FloatBeamWidth1)
{
    SizeType constexpr beamWidth{1};
    SamplingConfig const samplingConfig{beamWidth};

    TRY_LOG_EXCEPTION(testDecoder(nvinfer1::DataType::kFLOAT, samplingConfig));
}

TEST(GptDecoderBatchTest, FloatBeamWidth3)
{
    SizeType constexpr beamWidth{3};
    SamplingConfig const samplingConfig{beamWidth};

    TRY_LOG_EXCEPTION(testDecoder(nvinfer1::DataType::kFLOAT, samplingConfig));
}
