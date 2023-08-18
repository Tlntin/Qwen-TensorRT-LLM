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
#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class GptModelConfig
{
public:
    constexpr explicit GptModelConfig(SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize,
        nvinfer1::DataType dtype, bool useGptAttentionPlugin = false, bool useMultiQueryMode = false,
        bool inputPacked = false, bool useInflightBatchingGptAttentionPlugin = false)
        : mVocabSize(vocabSize)
        , mNbLayers(nbLayers)
        , mNbHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(useGptAttentionPlugin)
        , mUseInflightBatchingGptAttentionPlugin(useInflightBatchingGptAttentionPlugin)
        , mUseMultiQueryMode(useMultiQueryMode)
        , mInputPacked{inputPacked}
    {
    }

    [[nodiscard]] SizeType constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType constexpr getNbLayers() const noexcept
    {
        return mNbLayers;
    }

    [[nodiscard]] SizeType constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr useInflightBatchingGptAttentionPlugin() const noexcept
    {
        return mUseInflightBatchingGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr useMultiQueryMode() const noexcept
    {
        return mUseMultiQueryMode;
    }

    [[nodiscard]] bool constexpr isInputPacked() const noexcept
    {
        return mInputPacked;
    }

private:
    SizeType mVocabSize;
    SizeType mNbLayers;
    SizeType mNbHeads;
    SizeType mHiddenSize;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mUseInflightBatchingGptAttentionPlugin;
    bool mUseMultiQueryMode;
    bool mInputPacked;
};

} // namespace tensorrt_llm::runtime
