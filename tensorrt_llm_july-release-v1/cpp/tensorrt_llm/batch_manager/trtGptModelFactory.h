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

#include <memory>

#include "tensorrt_llm/batch_manager/trtGptModelInflightBatching.h"
#include "tensorrt_llm/batch_manager/trtGptModelV1.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/gptDecoderBatchInterface.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferPlugin.h>

namespace inflight_batcher
{
namespace batch_manager
{

enum class TrtGptModelType
{
    V1,
    InflightBatching
};

class TrtGptModelFactory
{
public:
    static std::vector<uint8_t> loadEngine(std::string const& enginePath)
    {
        std::ifstream engineFile(enginePath, std::ios::binary);
        TLLM_CHECK_WITH_INFO(engineFile.good(), std::string("Error opening engine file: " + enginePath));
        engineFile.seekg(0, std::ifstream::end);
        auto const size = engineFile.tellg();
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<uint8_t> engineBlob(size);
        engineFile.read(reinterpret_cast<char*>(engineBlob.data()), size);
        TLLM_CHECK_WITH_INFO(engineFile.good(), std::string("Error loading engine file: " + enginePath));
        return engineBlob;
    }

    // TODO: padId and endId should probably come from config

    static std::shared_ptr<TrtGptModel> create(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        int32_t maxSeqLen, int32_t maxNumRequests, int32_t beamWidth, int32_t endId = 50256, int32_t padId = 50256)
    {
        auto logger = std::make_shared<TllmLogger>();
        auto const json = GptJsonConfig::parse(trtEnginePath / "config.json");
        auto modelConfig = json.modelConfig();
        auto worldConfig = WorldConfig::mpi(*logger);

        auto const enginePath = trtEnginePath / json.engineFilename(worldConfig);
        auto const dtype = modelConfig.getDataType();

        auto engineBuffer = TrtGptModelFactory::loadEngine(enginePath);

        if (modelType == TrtGptModelType::V1)
        {
            return std::make_shared<TrtGptModelV1>(
                maxSeqLen, maxNumRequests, beamWidth, logger, modelConfig, worldConfig, engineBuffer);
        }
        else if (modelType == TrtGptModelType::InflightBatching)
        {
            auto runtime = std::make_shared<TllmRuntime>(engineBuffer.data(), engineBuffer.size(), *logger);
            std::shared_ptr<IGptDecoderBatch> gptDecoderBatch
                = std::make_shared<GptDecoderBatch>(modelConfig.getVocabSize(),
                    modelConfig.getVocabSizePadded(worldConfig.getSize()), endId, padId, runtime->getStreamPtr());

            return std::make_shared<TrtGptModelInflightBatching>(
                maxSeqLen, maxNumRequests, beamWidth, logger, modelConfig, worldConfig, runtime, gptDecoderBatch);
        }
        else
        {
            throw std::runtime_error("Invalid modelType in trtGptModelFactory");
        }
    }
};

} // namespace batch_manager
} // namespace inflight_batcher
