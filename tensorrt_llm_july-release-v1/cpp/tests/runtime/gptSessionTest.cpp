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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <algorithm>
#include <filesystem>

#include <NvInferPlugin.h>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace fs = std::filesystem;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const GPT_MODEL_PATH = TEST_RESOURCE_PATH / "models/rt_engine/gpt2";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
} // namespace

class GptSessionTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP();

        mLogger = std::make_shared<TllmLogger>();

        initLibNvInferPlugins(mLogger.get(), "tensorrt_llm");
    }

    void TearDown() override {}

    int mDeviceCount;
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

namespace
{
void testGptSession(fs::path const& modelPath, bool usePlugin, bool inputPacked, nvinfer1::DataType dtype,
    SizeType beamWidth, std::initializer_list<int> const& batchSizes, std::string const& resultsFile,
    std::shared_ptr<nvinfer1::ILogger> const& logger, bool replicateFirstBatch = false)
{
    ASSERT_TRUE(fs::exists(DATA_PATH));
    auto givenInput = tc::Tensor::loadNpy(DATA_PATH / "input_tokens.npy", tc::MEMORY_CPU);
    ASSERT_EQ(givenInput.shape.size(), 2);
    ASSERT_GT(givenInput.shape[0], 0);
    auto const nbGivenInputs = static_cast<SizeType>(givenInput.shape[0]);
    auto expectedOutput = tc::Tensor::loadNpy(DATA_PATH / resultsFile, tc::MEMORY_CPU);
    ASSERT_EQ(expectedOutput.shape.size(), 2);
    ASSERT_EQ(givenInput.shape[0] * beamWidth, expectedOutput.shape[0]);
    auto const givenInputData = givenInput.getPtr<int>();
    auto const expectedOutputData = expectedOutput.getPtr<int>();

    ASSERT_TRUE(fs::exists(modelPath));
    auto const json = GptJsonConfig::parse(modelPath / "config.json");
    auto const modelConfig = json.modelConfig();
    ASSERT_EQ(usePlugin, modelConfig.useGptAttentionPlugin() || modelConfig.useInflightBatchingGptAttentionPlugin());
    ASSERT_EQ(inputPacked, modelConfig.isInputPacked());
    auto const worldConfig = WorldConfig::mpi(*logger);
    auto const enginePath = modelPath / json.engineFilename(worldConfig);
    ASSERT_EQ(dtype, modelConfig.getDataType());

    auto const maxInputLength = static_cast<SizeType>(givenInput.shape[1]);
    auto const maxSeqLength = static_cast<SizeType>(expectedOutput.shape[1]);
    ASSERT_LT(maxInputLength, maxSeqLength);
    auto const maxNewTokens = maxSeqLength - maxInputLength;
    SamplingConfig samplingConfig{beamWidth};
    samplingConfig.temperature = std::vector{1.0f};
    samplingConfig.minLength = std::vector{1};
    samplingConfig.randomSeed = std::vector{42ull};
    samplingConfig.topK = std::vector{0};
    samplingConfig.topP = std::vector{0.0f};

    auto constexpr endId = 50256;
    auto constexpr padId = 50256;

    std::vector<SizeType> givenInputLengths(nbGivenInputs);
    for (SizeType i = 0; i < nbGivenInputs; ++i)
    {
        auto const seqBegin = givenInputData + i * maxInputLength;
        auto const it = std::find(seqBegin, seqBegin + maxInputLength, padId);
        givenInputLengths[i] = std::distance(seqBegin, it);
    }

    GptSession session{modelConfig, worldConfig, enginePath, logger};
    EXPECT_EQ(session.getDevice(), worldConfig.getDevice());
    // Use bufferManager for copying data to and from the GPU
    auto& bufferManager = session.getBufferManager();

    for (auto const batchSize : batchSizes)
    {
        session.setup(batchSize, maxInputLength, maxNewTokens, samplingConfig);

        // use 5 to 12 tokens from input
        std::vector<SizeType> inputLenghtsHost(batchSize);
        for (SizeType i = 0; i < batchSize; ++i)
        {
            const int inputIdx = replicateFirstBatch ? 0 : i % nbGivenInputs;
            inputLenghtsHost[i] = givenInputLengths[inputIdx];
        }
        auto inputLenghts = bufferManager.copyFrom(inputLenghtsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

        // copy inputs and wrap into shared_ptr
        GenerationInput::TensorPtr inputIds;
        if (inputPacked)
        {
            std::vector<SizeType> inputOffsetsHost(batchSize + 1);
            std::inclusive_scan(inputLenghtsHost.begin(), inputLenghtsHost.end(), inputOffsetsHost.begin() + 1);
            auto const totalInputSize = inputOffsetsHost.back();

            std::vector<std::int32_t> inputsHost(totalInputSize);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (replicateFirstBatch ? 0 : (i % nbGivenInputs) * maxInputLength);
                std::copy(seqBegin, seqBegin + inputLenghtsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
            }
            inputIds = bufferManager.copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);
        }
        else
        {
            std::vector<std::int32_t> inputsHost(batchSize * maxInputLength, padId);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto const seqBegin = givenInputData + (replicateFirstBatch ? 0 : (i % nbGivenInputs) * maxInputLength);
                std::copy(seqBegin, seqBegin + inputLenghtsHost[i], inputsHost.begin() + i * maxInputLength);
            }
            inputIds
                = bufferManager.copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
        }

        GenerationInput generationInput{endId, padId, std::move(inputIds), std::move(inputLenghts), inputPacked};
        generationInput.disableInputCopy = false;

        // runtime will allocate memory for output if this tensor is empty
        GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

        // repeat the same inputs multiple times for testing idempotency of `generate()`
        auto constexpr repetitions = 10;
        for (auto r = 0; r < repetitions; ++r)
        {
            SizeType numSteps = 0;
            generationOutput.onTokenGenerated
                = [&numSteps, maxNewTokens]([[maybe_unused]] GenerationOutput::TensorPtr const& outputIds,
                      [[maybe_unused]] SizeType step, bool finished)
            {
                ++numSteps;
                EXPECT_TRUE(!finished || numSteps == maxNewTokens);
            };
            session.generate(generationOutput, generationInput);
            EXPECT_EQ(numSteps, maxNewTokens);

            // compare outputs
            auto const& outputIds = generationOutput.ids;
            auto const& outputDims = outputIds->getShape();
            EXPECT_EQ(outputDims.nbDims, 3);
            EXPECT_EQ(outputDims.d[0], batchSize) << "r: " << r;
            EXPECT_EQ(outputDims.d[1], beamWidth) << "r: " << r;
            EXPECT_EQ(outputDims.d[2], maxSeqLength) << "r: " << r;
            auto outputHost = bufferManager.copyFrom(*outputIds, MemoryType::kCPU);
            auto output = bufferCast<std::int32_t>(*outputHost);
            bufferManager.getStream().synchronize();
            for (auto b = 0; b < batchSize; ++b)
            {
                for (auto beam = 0; beam < beamWidth; ++beam)
                {
                    bool anyMismatch = false;
                    for (auto i = 0; i < maxSeqLength; ++i)
                    {
                        auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                        const int expectedBatch = replicateFirstBatch ? 0 : b;
                        auto const expectIndex
                            = tc::flat_index2((expectedBatch % nbGivenInputs * beamWidth + beam), i, maxSeqLength);
                        EXPECT_EQ(output[outputIndex], expectedOutputData[expectIndex])
                            << " b: " << b << " beam: " << beam << " i: " << i;
                        anyMismatch |= (output[outputIndex] != expectedOutputData[expectIndex]);
                    }
                    ASSERT_FALSE(anyMismatch) << "batchSize: " << batchSize << ", r: " << r << ", b: " << b;
                }
            }

            // make sure to recreate the outputs in the next repetition
            outputIds->release();
        }
    }

    free(givenInputData);
    free(expectedOutputData);
}

auto constexpr kBatchSizes = {1, 8};

} // namespace

// Engines need to be generated using cpp/tests/resources/scripts/build_gpt_engines.py.
// Expected outputs need to be generated using cpp/tests/resources/scripts/generate_expected_output.py.

TEST_F(GptSessionTest, SamplingFP32)
{
    auto const modelPath{GPT_MODEL_PATH / "fp32-default/1-gpu"};
    bool constexpr usePlugin{false};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kFLOAT;
    SizeType constexpr beamWidth{1};
    testGptSession(
        modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes, "sampling/output_tokens_fp32.npy", mLogger);
}

TEST_F(GptSessionTest, SamplingFP32WithPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp32-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kFLOAT;
    SizeType constexpr beamWidth{1};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "sampling/output_tokens_fp32_plugin.npy", mLogger);
}

TEST_F(GptSessionTest, SamplingFP16)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-default/1-gpu"};
    bool constexpr usePlugin{false};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{1};
    testGptSession(
        modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes, "sampling/output_tokens_fp16.npy", mLogger);
}

TEST_F(GptSessionTest, SamplingFP16WithPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{1};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "sampling/output_tokens_fp16_plugin.npy", mLogger);
}

TEST_F(GptSessionTest, SamplingFP16WithPluginPacked)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-plugin-packed/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{true};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{1};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "sampling/output_tokens_fp16_plugin_packed.npy", mLogger);
}

TEST_F(GptSessionTest, BeamSearchFP32)
{
    GTEST_SKIP();

    auto const modelPath{GPT_MODEL_PATH / "fp32-default/1-gpu"};
    bool constexpr usePlugin{false};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kFLOAT;
    SizeType constexpr beamWidth{2};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp32.npy", mLogger);
}

TEST_F(GptSessionTest, BeamSearchFP32WithPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp32-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kFLOAT;
    SizeType constexpr beamWidth{2};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp32_plugin.npy", mLogger);
}

TEST_F(GptSessionTest, BeamSearchFP16)
{
    GTEST_SKIP();

    auto const modelPath{GPT_MODEL_PATH / "fp16-default/1-gpu"};
    bool constexpr usePlugin{false};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{2};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp16.npy", mLogger);
}

TEST_F(GptSessionTest, BeamSearchFP16WithPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{false};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{2};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp16_plugin.npy", mLogger);
}

TEST_F(GptSessionTest, BeamSearchFP16WithPluginPacked)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-plugin-packed/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{true};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{2};
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp16_plugin_packed.npy", mLogger);
}

TEST_F(GptSessionTest, FP16WithInflightBatchingSamplingPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-inflight-batching-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{true};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{1};
    // We replicate the first batch to have all sequences of the same length
    // Gpt Session does not support variable sequence length for In-flight batching plugin
    // Use gptSessionWithDecoderBatch instead
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "sampling/output_tokens_fp16_plugin_packed.npy", mLogger, true);
}

TEST_F(GptSessionTest, BeamSearchFP16WithInflightBatchingPlugin)
{
    auto const modelPath{GPT_MODEL_PATH / "fp16-inflight-batching-plugin/1-gpu"};
    bool constexpr usePlugin{true};
    bool constexpr inputPacked{true};
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    SizeType constexpr beamWidth{2};
    // We replicate the first batch to have all sequences of the same length
    // Gpt Session does not support variable sequence length for In-flight batching plugin
    // Use gptSessionWithDecoderBatch instead
    testGptSession(modelPath, usePlugin, inputPacked, dtype, beamWidth, kBatchSizes,
        "beam_search_2/output_tokens_fp16_plugin_packed.npy", mLogger, true);
}
