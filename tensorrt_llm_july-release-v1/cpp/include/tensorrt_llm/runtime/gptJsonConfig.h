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

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <filesystem>
#include <istream>
#include <string>
#include <utility>

namespace tensorrt_llm::runtime
{

class GptJsonConfig
{
public:
    class BuilderConfig
    {
    public:
        BuilderConfig(std::string precision, SizeType worldSize, SizeType numHeads, SizeType hiddenSize,
            SizeType vocabSize, SizeType numLayers, bool multiQueryMode, std::string name)
            : mPrecision(std::move(precision))
            , mWorldSize(worldSize)
            , mNumHeads(numHeads)
            , mHiddenSize(hiddenSize)
            , mVocabSize(vocabSize)
            , mNumLayers(numLayers)
            , mMultiQueryMode(multiQueryMode)
            , mName(std::move(name))
        {
        }

        [[nodiscard]] std::string precision() const
        {
            return mPrecision;
        }

        [[nodiscard]] SizeType worldSize() const
        {
            return mWorldSize;
        }

        [[nodiscard]] SizeType numHeads() const
        {
            return mNumHeads;
        }

        [[nodiscard]] SizeType hiddenSize() const
        {
            return mHiddenSize;
        }

        [[nodiscard]] SizeType vocabSize() const
        {
            return mVocabSize;
        }

        [[nodiscard]] SizeType numLayers() const
        {
            return mNumLayers;
        }

        [[nodiscard]] bool useMultiQueryMode() const
        {
            return mMultiQueryMode;
        }

        [[nodiscard]] std::string const& getName() const
        {
            return mName;
        }

    private:
        std::string mPrecision;
        SizeType mWorldSize;
        SizeType mNumHeads;
        SizeType mHiddenSize;
        SizeType mVocabSize;
        SizeType mNumLayers;
        bool mMultiQueryMode;
        std::string mName;
    };

    class PluginConfig
    {
    public:
        explicit PluginConfig(
            bool useGptAttentionPlugin, bool useInflightBatchingGptAttentionPlugin, bool removeInputPadding)
            : mUseGptAttentionPlugin{useGptAttentionPlugin}
            , mUseInflightBatchingGptAttentionPlugin{useInflightBatchingGptAttentionPlugin}
            , mRemoveInputPadding{removeInputPadding}
        {
        }

        [[nodiscard]] bool useGptAttentionPlugin() const
        {
            return mUseGptAttentionPlugin;
        }

        [[nodiscard]] bool useInflightBatchingGptAttentionPlugin() const
        {
            return mUseInflightBatchingGptAttentionPlugin;
        }

        [[nodiscard]] bool removeInputPadding() const
        {
            return mRemoveInputPadding;
        }

    private:
        bool mUseGptAttentionPlugin;
        bool mUseInflightBatchingGptAttentionPlugin;
        bool mRemoveInputPadding;
    };

    GptJsonConfig(BuilderConfig builderConfig, PluginConfig pluginConfig)
        : mBuilderConfig(std::move(builderConfig))
        , mPluginConfig(pluginConfig)
    {
    }

    static GptJsonConfig parse(std::string const& json);

    static GptJsonConfig parse(std::istream& json);

    static GptJsonConfig parse(std::filesystem::path const& path);

    [[nodiscard]] BuilderConfig const& builderConfig() const
    {
        return mBuilderConfig;
    }

    [[nodiscard]] PluginConfig const& pluginConfig() const
    {
        return mPluginConfig;
    }

    [[nodiscard]] GptModelConfig modelConfig() const;

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig, std::string const& model) const;

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig) const
    {
        return engineFilename(worldConfig, mBuilderConfig.getName());
    }

private:
    BuilderConfig const mBuilderConfig;
    PluginConfig const mPluginConfig;
};

} // namespace tensorrt_llm::runtime
