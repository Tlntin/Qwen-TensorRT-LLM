/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h"
#include "checkMacrosPlugin.h"
#include "gptAttentionCommon.h"
#include "gptAttentionCommon/gptAttentionCommonImpl.h"
#include "plugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::GPTAttentionPluginCreator;
using nvinfer1::plugin::GPTAttentionPlugin;

static const char* GPT_ATTENTION_PLUGIN_VERSION{"1"};
static const char* GPT_ATTENTION_PLUGIN_NAME{"GPTAttention"};

GPTAttentionPlugin::GPTAttentionPlugin(int num_heads, int head_size, int unidirectional, float q_scaling,
    int rotary_embedding_dim, bool neox_rotary_style, tensorrt_llm::kernels::ContextFMHAType context_fmha_type,
    bool multi_block_mode, bool multi_query_mode, bool int8_kv_cache, bool fp8_kv_cache, bool remove_input_padding,
    tensorrt_llm::kernels::AttentionMaskType mask_type, bool paged_kv_cache, nvinfer1::DataType type,
    bool in_flight_batching)
    : GPTAttentionPluginCommon(num_heads, head_size, unidirectional, q_scaling, rotary_embedding_dim, neox_rotary_style,
        context_fmha_type, multi_block_mode, multi_query_mode, int8_kv_cache, fp8_kv_cache, remove_input_padding,
        mask_type, paged_kv_cache, type)
    , mInFlightBatching(in_flight_batching)
{
    PLUGIN_ASSERT(!mInFlightBatching || mRemovePadding);
}

GPTAttentionPlugin::GPTAttentionPlugin(const void* data, size_t length)
    : GPTAttentionPluginCommon(data, GPTAttentionPluginCommon::getCommonSerializationSize())
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    d += GPTAttentionPluginCommon::getCommonSerializationSize();

    read(d, mInFlightBatching);
    PLUGIN_ASSERT(d == a + length);
    PLUGIN_ASSERT(!mInFlightBatching || mRemovePadding);
}

// IPluginV2DynamicExt Methods
GPTAttentionPlugin* GPTAttentionPlugin::clone() const noexcept
{
    return dynamic_cast<GPTAttentionPlugin*>(this->cloneImpl<GPTAttentionPlugin>());
}

// outputs (mMultiQueryMode == false)
//     output_tensor [batch_size, seq_len, local_hidden_size]
//     present_key_value_pool [blocks, 2, local_num_heads, tokens_per_block, head_size] if paged_kv_attention
//                         or [batch_size, 2, local_num_heads, max_seq_len, head_size]
// outputs (mMultiQueryMode == true)
//     output_tensor [batch_size, seq_len, local_hidden_size]
//     present_key_value_pool [blocks, 2, local_num_heads, tokens_per_block, head_size] if paged_kv_attention
//                         or [batch_size, 2, 1^, max_seq_len, head_size]
nvinfer1::DimsExprs GPTAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex == 0 || outputIndex == 1);
    if (outputIndex == 0)
    {
        auto ret = inputs[getInputTensorIdx()];
        ret.d[2] = exprBuilder.constant(mNumHeads * mHeadSize);
        return ret;
    }
    else
    {
        return inputs[outputIndex];
    }
}

bool GPTAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getSequenceLengthIdx() || pos == getPastKeyValueLengthIdx() || pos == getMaskedTokensIdx()
        || pos == getInputLengthsIdx() || pos == getMaxInputLengthIdx() || pos == getCacheIndirIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if ((mInt8KVCache || mFp8KVCache)
        && (pos == getKVCacheDequantizationScaleIdx() || pos == getKVCacheQuantizationScaleIdx()))
    {
        // int8_kv_scale for mType->int8 and int8->mType conversion
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mPagedKVCache && pos == getKVCacheBlockPointersIdx())
    {
        // pointers to kv cache blocks
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mInt8KVCache && (pos == getPastKeyValueIdx() || pos == nbInputs + 1))
    {
        // If use Int8 K/V cache we require I/O KV values to int8
        return (inOut[pos].type == nvinfer1::DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (mInFlightBatching && (pos == getHostInputLengthsIdx() || pos == getRequestTypesIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    return false;
}

void GPTAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GPTAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    const int max_input_length = inputs[getMaxInputLengthIdx()].dims.d[0];
    const int nbReq = inputs[getSequenceLengthIdx()].dims.d[0];
    auto const type = inputs[getInputTensorIdx()].type;
    size_t const context_workspace_size = getWorkspaceSizeForContext(type, nbReq, max_input_length);

    const int total_num_seq = inputs[getSequenceLengthIdx()].dims.d[0];
    size_t const generation_workspace_size = getWorkspaceSizeForGeneration(type, total_num_seq);

    return std::max(context_workspace_size, generation_workspace_size);
}

static int32_t getStride(nvinfer1::Dims const& dims, int n)
{
    PLUGIN_ASSERT(n >= 0 && n < dims.nbDims)
    return std::accumulate(dims.d + n + 1, dims.d + dims.nbDims, 1, std::multiplies<int32_t>{});
}

template <typename T, typename KVCacheBuffer>
int GPTAttentionPlugin::enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    int32_t const nbSeq = inputDesc[getInputLengthsIdx()].dims.d[0];
    if (!mInFlightBatching)
    {
        return enqueueSome<T, KVCacheBuffer>(0, nbSeq, 0, inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    // In-flight batching code path
    int32_t const beam_width = inputDesc[getCacheIndirIdx()].dims.d[1];
    // When using beam search, context requests must be followed by (beam_width - 1) none requests to reserve
    // cache/cache_indir space for later generation steps.
    PLUGIN_ASSERT(nbSeq % beam_width == 0);

    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getRequestTypesIdx()]);
    int32_t seqIdxBeg = 0;
    RequestType refReqType = reqTypes[0];
    int32_t tokenIdxBeg = 0;
    int32_t tokenIdxEnd = 0;
    // Find consecutive requests of the same type, and launch them in a batch.
    for (int32_t i = seqIdxBeg; i < nbSeq + 1; i++)
    {
        if (i < nbSeq && reqTypes[i] == refReqType)
        {
            tokenIdxEnd += (mRemovePadding ? static_cast<int32_t const*>(inputs[getHostInputLengthsIdx()])[i]
                                           : inputDesc[getInputTensorIdx()].dims.d[1]);
        }
        else
        {
            if (refReqType != RequestType::kNONE)
            {
                // When using beam search, context requests must be followed by (beam_width - 1) none requests to
                // reserve cache/cache_indir space for later generation steps. So this is always true for context or
                // generation request groups.
                PLUGIN_ASSERT(seqIdxBeg % beam_width == 0);
                enqueueSome<T, KVCacheBuffer>(
                    seqIdxBeg, i - seqIdxBeg, tokenIdxBeg, inputDesc, outputDesc, inputs, outputs, workspace, stream);
            }
            if (i < nbSeq)
            {
                seqIdxBeg = i;
                refReqType = reqTypes[i];
                tokenIdxBeg = tokenIdxEnd;
            }
        }
    }
    return 0;
}

template <typename T, typename KVCacheBuffer>
int GPTAttentionPlugin::enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg,
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    // inputs (mMultiQueryMode == false)
    //     input_tensor [batch_size, seq_len, local_hidden_size * 3]
    //                  [1, num_tokens, local_hidden_size * 3] when enable_remove_input_padding
    //     past_key_value_pool [blocks, 2, local_num_heads, tokens_per_block, head_size] if paged_kv_attention
    //                      or [batch_size, 2, local_num_heads, max_seq_len, head_size]
    //     sequence_length [batch_size]
    //     past_key_value_length [2] host scalars
    //          field 0: past_key_value_length
    //          field 1: is_context
    //     masked_tokens [batch_size, max_seq_len]
    //     input_lengths [batch_size]
    //     max_input_length [max_input_length]
    //     cache_indir [batch_beam / beam_width, beam_width, memory_max_len] (required in beamsearch)
    //     kv_cache_quantization_scale [1] (optional)
    //     kv_cache_dequantization_scale [1] (optional)
    //     block_pointers [batch_size, bead_width, 2, max_blocks_per_seq] (optional if paged kv cache)
    //     host_input_lengths [batch_size] int32. (optional, for in-flight batching)
    //     host_request_types [batch_size] int32. (optional, for in-flight batching) 0: context; 1: generation: 2: none
    //
    // outputs (mMultiQueryMode == false)
    //     output_tensor [batch_size, seq_len, local_hidden_size]
    //     present_key_value_pool [blocks, 2, local_num_heads, tokens_per_block, head_size] if paged_kv_attention
    //                         or [batch_size, 2, local_num_heads, max_seq_len, head_size]
    //
    // inputs (mMultiQueryMode == true; ^ marks the difference)
    //     input_tensor [batch_size, seq_len, local_hidden_size + 2 * head_size^]
    //                  [1, num_tokens, local_hidden_size + 2 * head_size^] when enable_remove_input_padding
    //     past_key_value_pool [blocks, 2, 1^, tokens_per_block, head_size] if paged_kv_attention
    //                      or [batch_size, 2, 1^, max_seq_len, head_size]
    //     sequence_length [batch_size]
    //     past_key_value_length [2] host scalars
    //          field 0: past_key_value_length
    //          field 1: is_context
    //     masked_tokens [batch_size, max_seq_len]
    //     input_lengths [batch_size]
    //     max_input_length [max_input_length]
    //     cache_indir [batch_beam / beam_width, beam_width, memory_max_len] (required in beamsearch)
    //     kv_cache_quantization_scale [1] (optional)
    //     kv_cache_dequantization_scale [1] (optional)
    //     block_pointers [batch_size, bead_width, 2, max_blocks_per_seq] (optional if paged kv cache)
    //     host_input_lengths [batch_size] int32. (optional, for in-flight batching)
    //     host_request_types [batch_size] int32. (optional, for in-flight batching) 0: context; 1: generation: 2: none
    //
    // outputs (mMultiQueryMode == true)
    //     output_tensor [batch_size, seq_len, local_hidden_size]
    //     present_key_value_pool [blocks, 2, 1^, tokens_per_block, head_size] if paged_kv_attention
    //                         or [batch_size, 2, 1^, max_seq_len, head_size]

    const T* attention_input
        = static_cast<const T*>(inputs[getInputTensorIdx()]) + inputDesc[getInputTensorIdx()].dims.d[2] * tokenIdxBeg;
    ;
    const int* sequence_length = static_cast<const int*>(inputs[getSequenceLengthIdx()]) + seqIdxBeg;

    const int* host_scalars = reinterpret_cast<const int*>(inputs[getPastKeyValueLengthIdx()]);
    int32_t const past_kv_len = host_scalars[0];
    bool is_context = false;
    if (mInFlightBatching)
    {
        auto const reqTypeInBatchPtr = static_cast<RequestType const*>(inputs[getRequestTypesIdx()]) + seqIdxBeg;
        is_context = (reqTypeInBatchPtr[0] == RequestType::kCONTEXT);
        PLUGIN_ASSERT(std::all_of(reqTypeInBatchPtr, reqTypeInBatchPtr + localNbSeq,
            [is_context](RequestType reqType)
            {
                PLUGIN_ASSERT(reqType == RequestType::kCONTEXT || reqType == RequestType::kGENERATION);
                return is_context == (reqType == RequestType::kCONTEXT);
            }));
    }
    else
    {
        is_context = static_cast<bool>(host_scalars[1]);
    }

    const int* masked_tokens = reinterpret_cast<const int*>(inputs[getMaskedTokensIdx()])
        + inputDesc[getMaskedTokensIdx()].dims.d[1] * seqIdxBeg;
    const int* input_lengths = reinterpret_cast<const int*>(inputs[getInputLengthsIdx()]) + seqIdxBeg;
    // We get max input length from the shape of a 1-D tensor
    int max_input_len = inputDesc[getMaxInputLengthIdx()].dims.d[0];

    if (mInFlightBatching)
    {
        PLUGIN_ASSERT(mRemovePadding)
        if (!is_context)
        {
            max_input_len = 1;
        }
        else
        {
            auto const host_input_lengths = static_cast<int32_t const*>(inputs[getHostInputLengthsIdx()]) + seqIdxBeg;
            auto const tmp = *std::max_element(host_input_lengths, host_input_lengths + localNbSeq);
            PLUGIN_ASSERT(tmp <= max_input_len);
            max_input_len = tmp;
        }
    }

    const float* kv_scale_orig_quant = nullptr;
    const float* kv_scale_quant_orig = nullptr;
    if (mInt8KVCache || mFp8KVCache)
    {
        assert(inputDesc[getKVCacheQuantizationScaleIdx()].type == DataType::kFLOAT);
        assert(inputDesc[getKVCacheDequantizationScaleIdx()].type == DataType::kFLOAT);
        kv_scale_orig_quant = reinterpret_cast<const float*>(inputs[getKVCacheQuantizationScaleIdx()]);
        kv_scale_quant_orig = reinterpret_cast<const float*>(inputs[getKVCacheDequantizationScaleIdx()]);
    }

    int max_blocks_per_sequence = 0;
    int tokens_per_block = 0;
    void* block_pointers = nullptr;
    if (mPagedKVCache)
    {
        // Div by 2 because we reinterpret int32 input as int64
        max_blocks_per_sequence = inputDesc[getKVCacheBlockPointersIdx()].dims.d[3] / 2;
        tokens_per_block = inputDesc[getPastKeyValueIdx()].dims.d[3];
        // Div by 2 because we reinterpret int32 input as int64
        void* const* const typed_block_pointers = static_cast<void* const*>(inputs[getKVCacheBlockPointersIdx()])
            + getStride(inputDesc[getKVCacheBlockPointersIdx()].dims, 0) / 2 * seqIdxBeg;
        block_pointers = const_cast<void*>(static_cast<void const*>(typed_block_pointers));
    }

    T* context_buf_ = (T*) (outputs[0]) + outputDesc[0].dims.d[2] * tokenIdxBeg;
    void* key_value_cache = nullptr;
    if (!mPagedKVCache)
    {
        auto const cacheElemSize = (mInt8KVCache ? 1 : sizeof(T));
        key_value_cache
            = static_cast<std::byte*>(outputs[1]) + cacheElemSize * getStride(outputDesc[1].dims, 0) * seqIdxBeg;
    }

    const int max_seq_len = inputDesc[getCacheIndirIdx()].dims.d[2];

    if (is_context) // context stage
    {
        const int batch_size = localNbSeq;
        const int request_batch_size = batch_size;
        const int request_seq_len = max_input_len;
        // num of total tokens (without paddings when remove paddings).
        int num_tokens = 0;
        if (!mRemovePadding)
        {
            num_tokens = request_batch_size * request_seq_len;
        }
        else if (mInFlightBatching)
        {
            auto const host_input_lengths = static_cast<int32_t const*>(inputs[getHostInputLengthsIdx()]) + seqIdxBeg;
            num_tokens = std::accumulate(host_input_lengths, host_input_lengths + localNbSeq, 0);
        }
        else
        {
            num_tokens = inputDesc[getInputTensorIdx()].dims.d[1];
        }

        enqueueContext<T, KVCacheBuffer>(attention_input, max_input_len, max_seq_len, input_lengths,
            kv_scale_orig_quant, kv_scale_quant_orig, context_buf_, key_value_cache, block_pointers, batch_size,
            num_tokens, tokens_per_block, max_blocks_per_sequence, workspace, stream);
    }
    else // generation stage; input_seq_len == 1
    {
        int batch_beam = localNbSeq;
        const int beam_width = inputDesc[getCacheIndirIdx()].dims.d[1];
        PLUGIN_ASSERT(batch_beam % beam_width == 0);
        int32_t const num_requests = batch_beam / beam_width;
        PLUGIN_ASSERT(seqIdxBeg % beam_width == 0);
        int32_t const reqIdxBeg = seqIdxBeg / beam_width;
        const int* cache_indir = beam_width == 1 ? nullptr
                                                 : reinterpret_cast<const int*>(inputs[getCacheIndirIdx()])
                + reqIdxBeg * getStride(inputDesc[getCacheIndirIdx()].dims, 0);
        enqueueGeneration<T, KVCacheBuffer>(attention_input, max_input_len, sequence_length, past_kv_len, beam_width,
            masked_tokens, input_lengths, kv_scale_orig_quant, kv_scale_quant_orig, context_buf_, key_value_cache,
            block_pointers, max_seq_len, num_requests, tokens_per_block, max_blocks_per_sequence, cache_indir,
            workspace, stream);
    }
    return 0;
}

template <typename T>
int GPTAttentionPlugin::enqueueDispatchKVCacheType(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    if (mPagedKVCache)
    {
        return enqueueImpl<T, KVBlockArray>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        return enqueueImpl<T, KVLinearBuffer>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 0;
}

int GPTAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueDispatchKVCacheType<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueDispatchKVCacheType<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueDispatchKVCacheType<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GPTAttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0 || index == 1);
    return inputTypes[index];
}

// IPluginV2 Methods

const char* GPTAttentionPlugin::getPluginType() const noexcept
{
    return GPT_ATTENTION_PLUGIN_NAME;
}

const char* GPTAttentionPlugin::getPluginVersion() const noexcept
{
    return GPT_ATTENTION_PLUGIN_VERSION;
}

int GPTAttentionPlugin::getNbOutputs() const noexcept
{
    return 2;
}

size_t GPTAttentionPlugin::getSerializationSize() const noexcept
{
    return GPTAttentionPluginCommon::getCommonSerializationSize() + sizeof(mInFlightBatching);
}

void GPTAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    GPTAttentionPluginCommon::serializeCommon(buffer);
    d += GPTAttentionPluginCommon::getCommonSerializationSize();
    write(d, mInFlightBatching);
    assert(d == a + getSerializationSize());
}

///////////////

GPTAttentionPluginCreator::GPTAttentionPluginCreator()
    : GPTAttentionPluginCreatorCommon()
{

    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GPTAttentionPluginCreator::getPluginName() const noexcept
{
    return GPT_ATTENTION_PLUGIN_NAME;
}

const char* GPTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return GPT_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* GPTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GPTAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    PluginFieldParser p{fc->nbFields, fc->fields};

    try
    {
        auto* obj = new GPTAttentionPlugin(p.getScalar<int32_t>("num_heads").value(),
            p.getScalar<int32_t>("head_size").value(), p.getScalar<int32_t>("unidirectional").value(),
            p.getScalar<float>("q_scaling").value(), p.getScalar<int32_t>("rotary_embedding_dim").value(),
            static_cast<bool>(p.getScalar<int8_t>("neox_rotary_style").value()),
            static_cast<ContextFMHAType>(p.getScalar<int8_t>("context_fmha_type").value()),
            static_cast<bool>(p.getScalar<int8_t>("multi_block_mode").value()),
            static_cast<bool>(p.getScalar<int8_t>("multi_query_mode").value()),
            static_cast<bool>(p.getScalar<int32_t>("int8_kv_cache").value()),
            static_cast<bool>(p.getScalar<int32_t>("fp8_kv_cache").value()),
            static_cast<bool>(p.getScalar<int8_t>("remove_input_padding").value()),
            static_cast<AttentionMaskType>(p.getScalar<int32_t>("mask_type").value()),
            static_cast<bool>(p.getScalar<int32_t>("paged_kv_cache").value()),
            static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            p.getScalar<int32_t>("in_flight_batching").value());
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GPTAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GPTAttentionPlugin::destroy()
    try
    {
        auto* obj = new GPTAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GPTAttentionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GPTAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
