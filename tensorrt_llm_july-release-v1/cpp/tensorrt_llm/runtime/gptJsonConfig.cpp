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

#include "tensorrt_llm/runtime/gptJsonConfig.h"

#include "tensorrt_llm/common/assert.h"

#include <fstream>
#include <nlohmann/json.hpp>

using namespace tensorrt_llm::runtime;

namespace
{

template <typename InputType>
GptJsonConfig parseJson(InputType&& i)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    auto json = nlohmann::json::parse(i, nullptr, allowExceptions, ingoreComments);
    auto const& mBuilderConfig = json.at("builder_config");
    auto const& pluginConfig = json.at("plugin_config");
    auto const& gptAttentionPlugin = pluginConfig.at("gpt_attention_plugin");
    auto const& inflightBatchingGptAttentionPlugin = pluginConfig.at("inflight_batching_gpt_attention_plugin");
    return GptJsonConfig{GptJsonConfig::BuilderConfig{mBuilderConfig.at("precision").template get<std::string>(),
                             mBuilderConfig.at("tensor_parallel").template get<SizeType>(),
                             mBuilderConfig.at("num_heads").template get<SizeType>(),
                             mBuilderConfig.at("hidden_size").template get<SizeType>(),
                             mBuilderConfig.at("vocab_size").template get<SizeType>(),
                             mBuilderConfig.at("num_layers").template get<SizeType>(),
                             mBuilderConfig.at("multi_query_mode").template get<bool>(),
                             mBuilderConfig.at("name").template get<std::string>()},
        GptJsonConfig::PluginConfig{!gptAttentionPlugin.is_boolean() || gptAttentionPlugin.template get<bool>(),
            !inflightBatchingGptAttentionPlugin.is_boolean() || inflightBatchingGptAttentionPlugin.template get<bool>(),
            pluginConfig.at("remove_input_padding").template get<bool>()}};
}

} // namespace

std::string GptJsonConfig::engineFilename(WorldConfig const& worldConfig, std::string const& model) const

{
    TLLM_CHECK_WITH_INFO(mBuilderConfig.worldSize() == mBuilderConfig.worldSize(), "world size mismatch");
    return model + "_" + mBuilderConfig.precision() + "_tp" + std::to_string(worldConfig.getSize()) + "_rank"
        + std::to_string(worldConfig.getRank()) + ".engine";
}

GptModelConfig GptJsonConfig::modelConfig() const
{
    auto const& bc = builderConfig();
    auto const worldSize = bc.worldSize();
    auto const numHeads = bc.numHeads() / worldSize;
    auto const hiddenSize = bc.hiddenSize() / worldSize;
    auto const vocabSize = bc.vocabSize();
    auto const numLayers = bc.numLayers();
    auto const dataType = (bc.precision().compare("float16")) ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF;
    auto const useMultiQueryMode = bc.useMultiQueryMode();

    auto const& pc = pluginConfig();
    auto const useGptAttentionPlugin = pc.useGptAttentionPlugin();
    auto const useInflightBatchingGptAttentionPlugin = pc.useInflightBatchingGptAttentionPlugin();
    auto const removeInputPadding = pc.removeInputPadding();

    return GptModelConfig{vocabSize, numLayers, numHeads, hiddenSize, dataType, useGptAttentionPlugin,
        useMultiQueryMode, removeInputPadding, useInflightBatchingGptAttentionPlugin};
}

GptJsonConfig GptJsonConfig::parse(std::string const& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::istream& json)
{
    return parseJson(json);
}

GptJsonConfig GptJsonConfig::parse(std::filesystem::path const& path)
{
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(path), std::string("File does not exist: ") + path.string());
    std::ifstream json(path);
    return parse(json);
}
