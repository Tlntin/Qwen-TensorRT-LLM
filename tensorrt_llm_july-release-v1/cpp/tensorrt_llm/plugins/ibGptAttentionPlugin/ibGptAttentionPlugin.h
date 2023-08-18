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
#ifndef TRT_IB_GPT_ATTENTION_PLUGIN_H
#define TRT_IB_GPT_ATTENTION_PLUGIN_H
#include "NvInferPlugin.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <cassert>
#include <cstdint>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

// The main purpose is to define copy behavior for streams and events so we can use default ctor for plugin.
class MultiStreamResource
{
public:
    MultiStreamResource(size_t nbStreams)
        : mStreams(nbStreams)
    {
    }

    MultiStreamResource(MultiStreamResource const& src)
        : MultiStreamResource(src.mStreams.size())
    {
    }

protected:
    std::vector<tensorrt_llm::runtime::CudaStream> mStreams;

    struct EvDel
    {
        void operator()(cudaEvent_t ev) const noexcept
        {
            auto const err = cudaEventDestroy(ev);
            if (err != cudaSuccess)
            {
                nvinfer1::plugin::logError(cudaGetErrorString(err), __FILE__, FN_NAME, __LINE__);
            }
        }
    };

    std::unique_ptr<CUevent_st, EvDel> mEvent = []()
    {
        cudaEvent_t ev{};
        TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        return std::unique_ptr<CUevent_st, EvDel>{ev};
    }();
};

// clang-format off
// inputs
//0     input_tensor [1, total_num_tokens, local_hidden_size * 3] or [1, total_num_tokens, local_hidden_size + 2 * head_size] (MQA)
//1     past_key_value [blocks, 2, local_num_heads, tokens_per_block, head_size] or [batch_size * beam_width, 2, local_num_heads, max_seq_len, head_size]
//2     host_beam_width_list [nbReq] int32
//3     host_input_lengths [nbReq] int32
//4     input_lengths [nbSeq] int32
//5     past_key_value_pointers [nbReq, 2] host int32. Each row (int32*2) is a pointer (in little-endian) to device memory buffer with shape [beam_width, local_num_heads(or 1 for MQA), max_seq_len, head_size]
//6     host_past_key_value_lengths [nbReq] int32
//7     cache_indir_pointers [nbReq, 2] host int32. Each row (int32*2) is a pointer (in little-endian) to device memory buffer with shape [beam_width, memory_max_len] (required in beamsearch). memory_max_len is required to be equal to cache_max_seq_len (see next input) for now.
//8     host_req_cache_max_seq_lengths [nbReq] int32.
//9     kv_cache_quantization_scale [1] (optional)
//10    kv_cache_dequantization_scale [1] (optional)
//11    pointers_to_kv_cache_block_pointers [nbReq, 2] (optional)
//12    kv_cache_block_pointers [nbReq, beam_width, 2, maxBlocksPerSeq] (optional)
//
// outputs
//     output_tensor [1, total_num_tokens, local_hidden_size] or [1, total_num_tokens, local_hidden_size] (MQA)
//     past_key_value [blocks, 2, local_num_heads, tokens_per_block, head_size] or [batch_size * beam_width, 2, local_num_heads, max_seq_len, head_size]
// clang-format on
class IBGPTAttentionPlugin : public GPTAttentionPluginCommon, public MultiStreamResource
{
public:
    IBGPTAttentionPlugin() = delete;

    IBGPTAttentionPlugin(int num_heads, int head_size, int unidirectional, float q_scaling, int rotary_embedding_dim,
        bool neox_rotary_style, tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode,
        bool multi_query_mode, bool int8_kv_cache, bool fp8_kv_cache,
        tensorrt_llm::kernels::AttentionMaskType mask_type, nvinfer1::DataType type, int max_input_len,
        int max_beam_width, bool paged_kv_cache);

    IBGPTAttentionPlugin(const void* data, size_t length);

    ~IBGPTAttentionPlugin() override = default;

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
    int enqueueImpl(int32_t reqIdx, int32_t seqOffset, int32_t inputOffset, const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream);

    template <typename T>
    int enqueueDispatchKVCacheType(int32_t reqIdx, int32_t seqOffset, int32_t inputOffset,
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    IBGPTAttentionPlugin* clone() const noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

private:
    void setMaxNbReq(int32_t maxNbReq);
    int getMaxSeqLenTile(int elemSize, int input_length) const;
    size_t getWorkspaceSizeForOneReq(DataType type, int32_t input_seq_len, int32_t beam_width) const noexcept;

    size_t getLocalSerializationSize() const noexcept;

    inline int getInputTensorIdx() const
    {
        return 0;
    }

    inline int getPastKeyValueIdx() const
    {
        return 1;
    }

    inline int getHostBeamWidthsIdx() const
    {
        return 2;
    }

    inline int getHostInputLengthsIdx() const
    {
        return 3;
    }

    inline int getInputLengthsIdx() const
    {
        return 4;
    }

    inline int getPastKeyValuePointersIdx() const
    {
        return 5;
    }

    inline int getHostPastKeyValueLengthsIdx() const
    {
        return 6;
    }

    inline int getCacheIndirectionPointersIdx() const
    {
        return 7;
    }

    inline int getHostReqCacheMaxSeqLengthsIdx() const
    {
        return 8;
    }

    inline int getKVCacheQuantizationScaleIdx() const
    {
        return 9;
    }

    inline int getKVCacheDequantizationScaleIdx() const
    {
        return 10;
    }

    inline int getPointersToKVCacheBlockPointersIdx() const
    {
        return (mInt8KVCache || mFp8KVCache) ? 11 : 9;
    }

    inline int getKVCacheBlockPointersIdx() const
    {
        return (mInt8KVCache || mFp8KVCache) ? 12 : 10;
    }

private:
    int mMaxNbReq = 0;
    int mMaxBeamWidth;
    int mMaxInputLength;
};

class IBGPTAttentionPluginCreator : public GPTAttentionPluginCreatorCommon
{
public:
    IBGPTAttentionPluginCreator();

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

#endif // TRT_IB_GPT_ATTENTION_PLUGIN_H
