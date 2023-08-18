/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#ifndef TRT_GPT_ATTENTION_PLUGIN_H
#define TRT_GPT_ATTENTION_PLUGIN_H
#include "NvInferPlugin.h"
#include "checkMacrosPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class GPTAttentionPlugin : public GPTAttentionPluginCommon
{
public:
    GPTAttentionPlugin(int num_heads, int head_size, int unidirectional, float q_scaling, int rotary_embedding_dim,
        bool neox_rotary_style, tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode,
        bool multi_query_mode, bool int8_kv_cache, bool fp8_kv_cache, bool remove_input_padding,
        tensorrt_llm::kernels::AttentionMaskType mask_type, bool paged_kv_cache, nvinfer1::DataType type,
        bool in_flight_batching);

    GPTAttentionPlugin(const void* data, size_t length);

    ~GPTAttentionPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    template <typename T, typename KVCacheBuffer>
    int enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    template <typename T>
    int enqueueDispatchKVCacheType(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    GPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    enum class RequestType : int32_t
    {
        kCONTEXT,
        kGENERATION,
        kNONE
    };

private:
    bool mInFlightBatching = false;

private:
    template <typename T, typename KVCacheBuffer>
    int enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg,
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    inline int getInputTensorIdx() const
    {
        return 0;
    }

    inline int getPastKeyValueIdx() const
    {
        return 1;
    }

    inline int getSequenceLengthIdx() const
    {
        return 2;
    }

    inline int getPastKeyValueLengthIdx() const
    {
        return 3;
    }

    inline int getMaskedTokensIdx() const
    {
        return 4;
    }

    inline int getInputLengthsIdx() const
    {
        return 5;
    }

    inline int getMaxInputLengthIdx() const
    {
        return 6;
    }

    inline int getCacheIndirIdx() const
    {
        return 7;
    }

    inline int getKVCacheQuantizationScaleIdx() const
    {
        return 8;
    }

    inline int getKVCacheDequantizationScaleIdx() const
    {
        return 9;
    }

    inline int getKVCacheBlockPointersIdx() const
    {
        return mInt8KVCache ? 10 : 8;
    }

    int32_t getHostInputLengthsIdx() const
    {
        PLUGIN_ASSERT(mInFlightBatching);
        return (mInt8KVCache ? 10 : 8) + (mPagedKVCache ? 1 : 0);
    }

    int32_t getRequestTypesIdx() const
    {
        return getHostInputLengthsIdx() + 1;
    }
};

class GPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    GPTAttentionPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_GPT_ATTENTION_PLUGIN_H
