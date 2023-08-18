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
#include "tensorrt_llm/plugins/ibGptAttentionPlugin/ibGptAttentionPlugin.h"
#include "checkMacrosPlugin.h"
#include "gptAttentionCommon/gptAttentionCommonImpl.h"
#include "plugin.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <algorithm>
#include <cstdint>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::IBGPTAttentionPluginCreator;
using nvinfer1::plugin::IBGPTAttentionPlugin;

static const char* IB_GPT_ATTENTION_PLUGIN_VERSION{"1"};
static const char* IB_GPT_ATTENTION_PLUGIN_NAME{"IBGPTAttention"};

IBGPTAttentionPlugin::IBGPTAttentionPlugin(int num_heads, int head_size, int unidirectional, float q_scaling,
    int rotary_embedding_dim, bool neox_rotary_style, ContextFMHAType context_fmha_type, bool multi_block_mode,
    bool multi_query_mode, bool int8_kv_cache, bool fp8_kv_cache, AttentionMaskType mask_type, nvinfer1::DataType type,
    int max_input_len, int max_beam_width, bool paged_kv_cache)
    : GPTAttentionPluginCommon(num_heads, head_size, unidirectional, q_scaling, rotary_embedding_dim, neox_rotary_style,
        context_fmha_type, multi_block_mode, multi_query_mode, int8_kv_cache, fp8_kv_cache, true, mask_type,
        paged_kv_cache, type)
    , MultiStreamResource(0)
    , mMaxBeamWidth(max_beam_width)
    , mMaxInputLength(max_input_len)
{
}

// Parameterized constructor
IBGPTAttentionPlugin::IBGPTAttentionPlugin(const void* data, size_t length)
    : GPTAttentionPluginCommon(data, length - getLocalSerializationSize())
    , MultiStreamResource(0)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    d += GPTAttentionPluginCommon::getCommonSerializationSize();

    read(d, mMaxNbReq);
    read(d, mMaxBeamWidth);
    read(d, mMaxInputLength);
    PLUGIN_ASSERT(d == a + length);
    setMaxNbReq(mMaxNbReq);
}

// IPluginV2DynamicExt Methods
IBGPTAttentionPlugin* IBGPTAttentionPlugin::clone() const noexcept
{
    return dynamic_cast<IBGPTAttentionPlugin*>(this->cloneImpl<IBGPTAttentionPlugin>());
}

// outputs
//     output_tensor [batch_size, seq_len, local_hidden_size]
nvinfer1::DimsExprs IBGPTAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(outputIndex < 2);
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

bool IBGPTAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostBeamWidthsIdx() || pos == getHostInputLengthsIdx() || pos == getInputLengthsIdx()
        || pos == getHostPastKeyValueLengthsIdx() || pos == getCacheIndirectionPointersIdx()
        || pos == getHostReqCacheMaxSeqLengthsIdx() || pos == getPastKeyValuePointersIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mPagedKVCache && (pos == getKVCacheBlockPointersIdx() || pos == getPointersToKVCacheBlockPointersIdx()))
    {
        // pointers to kv cache blocks
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if ((mInt8KVCache || mFp8KVCache)
        && (pos == getKVCacheDequantizationScaleIdx() || pos == getKVCacheQuantizationScaleIdx()))
    {
        // int8_kv_scale for mType->int8 and int8->mType conversion
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mInt8KVCache && (pos == getPastKeyValueIdx() || pos == nbInputs + 1))
    {
        // If use Int8 K/V cache we require I/O KV values to int8
        return (inOut[pos].type == nvinfer1::DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void IBGPTAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    setMaxNbReq(in[getHostBeamWidthsIdx()].max.d[0]);
}

size_t IBGPTAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    auto const type = inputs[getInputTensorIdx()].type;
    auto const nbReq = inputs[getHostBeamWidthsIdx()].dims.d[0];
    return getWorkspaceSizeForOneReq(type, mMaxInputLength, mMaxBeamWidth) * nbReq;
}

size_t IBGPTAttentionPlugin::getWorkspaceSizeForOneReq(
    DataType type, int32_t input_seq_len, int32_t beam_width) const noexcept
{

    size_t const context_workspace_size = getWorkspaceSizeForContext(type, 1, mMaxInputLength);
    size_t const generation_workspace_size = getWorkspaceSizeForGeneration(type, beam_width);

    return std::max(context_workspace_size, generation_workspace_size);
}

void IBGPTAttentionPlugin::setMaxNbReq(int32_t maxNbReq)
{
    mMaxNbReq = maxNbReq;
    mStreams.resize(mMaxNbReq);
}

template <typename T, typename KVCacheBuffer>
int IBGPTAttentionPlugin::enqueueImpl(int32_t reqIdx, int32_t seqOffset, int32_t inputOffset,
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    const T* attention_input
        = static_cast<const T*>(inputs[getInputTensorIdx()]) + inputDesc[getInputTensorIdx()].dims.d[2] * inputOffset;
    const int* sequence_length = nullptr;

    int32_t const past_kv_len = static_cast<int32_t const*>(inputs[getHostPastKeyValueLengthsIdx()])[reqIdx];
    const int input_seq_len = static_cast<int32_t const*>(inputs[getHostInputLengthsIdx()])[reqIdx];
    int32_t const beam_width = static_cast<int32_t const*>(inputs[getHostBeamWidthsIdx()])[reqIdx];
    // Currently we don't support context steps with cache. But if we do in the future, it's also OK to use generation
    // code path (with beam_width == 1) to handle context step with input_seq_len == 1.
    const bool is_context = (past_kv_len == 0 && beam_width == 1 && input_seq_len > 1);

    const int* masked_tokens = nullptr;
    const int* input_lengths = static_cast<const int*>(inputs[getInputLengthsIdx()]) + seqOffset;

    auto const getPtr = [](void const* ptrList, int32_t idx)
    {
        auto const& buf = static_cast<uint32_t const(*)[2]>(ptrList)[idx];
        static_assert(sizeof(std::uintptr_t) == 8);
        // portable implementation for pointer stored as uint32_t[2] with little endian.
        return reinterpret_cast<void*>(std::uintptr_t{buf[0]} | (std::uintptr_t{buf[1]} << 32));
    };
    auto const cache_indir = static_cast<int const*>(getPtr(inputs[getCacheIndirectionPointersIdx()], reqIdx));

    const float* kv_scale_orig_quant = nullptr;
    const float* kv_scale_quant_orig = nullptr;
    if (mInt8KVCache || mFp8KVCache)
    {
        assert(inputDesc[getKVCacheQuantizationScaleIdx()].type == DataType::kFLOAT);
        assert(inputDesc[getKVCacheDequantizationScaleIdx()].type == DataType::kFLOAT);
        kv_scale_orig_quant = static_cast<const float*>(inputs[getKVCacheQuantizationScaleIdx()]);
        kv_scale_quant_orig = static_cast<const float*>(inputs[getKVCacheDequantizationScaleIdx()]);
    }

    int max_blocks_per_sequence = 0;
    int tokens_per_block = 0;
    void* block_pointers = nullptr;
    if (mPagedKVCache)
    {
        // Div by 2 because we reinterpret int32 input as int64
        max_blocks_per_sequence = inputDesc[getKVCacheBlockPointersIdx()].dims.d[3] / 2;
        tokens_per_block = inputDesc[getPastKeyValueIdx()].dims.d[3];
        block_pointers = const_cast<void*>(getPtr(inputs[getPointersToKVCacheBlockPointersIdx()], reqIdx));
    }

    T* context_buf_ = static_cast<T*>(outputs[0]) + outputDesc[0].dims.d[2] * inputOffset;
    T* key_value_cache = static_cast<T*>(getPtr(inputs[getPastKeyValuePointersIdx()], reqIdx));

    const int max_seq_len = static_cast<int32_t const*>(inputs[getHostReqCacheMaxSeqLengthsIdx()])[reqIdx];

    if (is_context) // context stage
    {
        const int batch_size = 1;
        const int request_batch_size = batch_size;
        const int request_seq_len = input_seq_len;
        // num of total tokens (without paddings when remove paddings).
        const int num_tokens = mRemovePadding ? beam_width * request_seq_len : request_batch_size * request_seq_len;
        enqueueContext<T, KVCacheBuffer>(attention_input, input_seq_len, max_seq_len, input_lengths,
            kv_scale_orig_quant, kv_scale_quant_orig, context_buf_, key_value_cache, block_pointers, batch_size,
            num_tokens, tokens_per_block, max_blocks_per_sequence, workspace, stream);
    }
    else // generation stage; input_seq_len == 1
    {
        int32_t const num_requests = 1;
        enqueueGeneration<T, KVCacheBuffer>(attention_input, 1, sequence_length, past_kv_len, beam_width, masked_tokens,
            input_lengths, kv_scale_orig_quant, kv_scale_quant_orig, context_buf_, key_value_cache, block_pointers,
            max_seq_len, num_requests, tokens_per_block, max_blocks_per_sequence, cache_indir, workspace, stream);
    }

    return 0;
}

template <typename T>
int IBGPTAttentionPlugin::enqueueDispatchKVCacheType(int32_t reqIdx, int32_t seqOffset, int32_t inputOffset,
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    if (mPagedKVCache)
    {
        return enqueueImpl<T, KVBlockArray>(
            reqIdx, seqOffset, inputOffset, inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        return enqueueImpl<T, KVLinearBuffer>(
            reqIdx, seqOffset, inputOffset, inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 0;
}

int IBGPTAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // streams in mStreams will wait for this event
    PLUGIN_CUASSERT(cudaEventRecord(mEvent.get(), stream));
    auto const type = inputDesc[getInputTensorIdx()].type;
    auto const nbReq = inputDesc[getHostBeamWidthsIdx()].dims.d[0];
    std::byte* nextWorkspace = static_cast<std::byte*>(workspace);
    int32_t seqOffset = 0;
    int32_t inputOffset = 0;
    auto const host_beam_width_list = static_cast<int32_t const*>(inputs[getHostBeamWidthsIdx()]);
    auto const host_input_lengths = static_cast<int32_t const*>(inputs[getHostInputLengthsIdx()]);
    // mStreams.at(i) will wait for `stream`, then kernels for each request is launched into mStreams.at(i)
    for (int32_t i = 0; i < nbReq; i++)
    {
        int32_t const beamWidth = host_beam_width_list[i];
        cudaStream_t st = mStreams.at(i).get();
        PLUGIN_CUASSERT(cudaStreamWaitEvent(st, mEvent.get()));

        switch (type)
        {
        case DataType::kHALF:
            enqueueDispatchKVCacheType<half>(
                i, seqOffset, inputOffset, inputDesc, outputDesc, inputs, outputs, nextWorkspace, st);
            break;
        case DataType::kFLOAT:
            enqueueDispatchKVCacheType<float>(
                i, seqOffset, inputOffset, inputDesc, outputDesc, inputs, outputs, nextWorkspace, st);
            break;
#ifdef ENABLE_BF16
        case DataType::kBF16:
            return enqueueDispatchKVCacheType<__nv_bfloat16>(
                i, seqOffset, inputOffset, inputDesc, outputDesc, inputs, outputs, nextWorkspace, st);
            break;
#endif
        default: PLUGIN_FAIL("not implemented");
        }

        auto const inputLength = host_input_lengths[i];
        nextWorkspace += getWorkspaceSizeForOneReq(type, inputLength, beamWidth);
        seqOffset += beamWidth;
        inputOffset += inputLength * beamWidth;
    }
    // let `stream` wait for all the mStreams
    for (int32_t i = 0; i < nbReq; i++)
    {
        PLUGIN_CUASSERT(cudaEventRecord(mEvent.get(), mStreams.at(i).get()));
        PLUGIN_CUASSERT(cudaStreamWaitEvent(stream, mEvent.get()));
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType IBGPTAttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0 || index == 1);
    return inputTypes[index];
}

// IPluginV2 Methods

const char* IBGPTAttentionPlugin::getPluginType() const noexcept
{
    return IB_GPT_ATTENTION_PLUGIN_NAME;
}

const char* IBGPTAttentionPlugin::getPluginVersion() const noexcept
{
    return IB_GPT_ATTENTION_PLUGIN_VERSION;
}

int IBGPTAttentionPlugin::getNbOutputs() const noexcept
{
    return 2;
}

size_t IBGPTAttentionPlugin::getLocalSerializationSize() const noexcept
{
    return sizeof(mMaxNbReq) + sizeof(mMaxBeamWidth) + sizeof(mMaxInputLength);
}

size_t IBGPTAttentionPlugin::getSerializationSize() const noexcept
{
    return GPTAttentionPluginCommon::getCommonSerializationSize() + getLocalSerializationSize();
}

void IBGPTAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    GPTAttentionPluginCommon::serializeCommon(buffer);
    d += GPTAttentionPluginCommon::getCommonSerializationSize();
    write(d, mMaxNbReq);
    write(d, mMaxBeamWidth);
    write(d, mMaxInputLength);
    assert(d == a + getSerializationSize());
}

///////////////

IBGPTAttentionPluginCreator::IBGPTAttentionPluginCreator()
    : GPTAttentionPluginCreatorCommon()
{
    mPluginAttributes.emplace_back(PluginField("max_input_len", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("max_beam_width", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* IBGPTAttentionPluginCreator::getPluginName() const noexcept
{
    return IB_GPT_ATTENTION_PLUGIN_NAME;
}

const char* IBGPTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return IB_GPT_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* IBGPTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* IBGPTAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    PluginFieldParser p{fc->nbFields, fc->fields};

    try
    {
        auto* obj = new IBGPTAttentionPlugin(p.getScalar<int32_t>("num_heads").value(),
            p.getScalar<int32_t>("head_size").value(), p.getScalar<int32_t>("unidirectional").value(),
            p.getScalar<float>("q_scaling").value(), p.getScalar<int32_t>("rotary_embedding_dim").value(),
            static_cast<bool>(p.getScalar<int8_t>("neox_rotary_style").value()),
            static_cast<ContextFMHAType>(p.getScalar<int8_t>("context_fmha_type").value()),
            static_cast<bool>(p.getScalar<int8_t>("multi_block_mode").value()),
            static_cast<bool>(p.getScalar<int8_t>("multi_query_mode").value()),
            static_cast<bool>(p.getScalar<int32_t>("int8_kv_cache").value()),
            static_cast<bool>(p.getScalar<int32_t>("fp8_kv_cache").value()),
            static_cast<AttentionMaskType>(p.getScalar<int32_t>("mask_type").value()),
            static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            p.getScalar<int32_t>("max_input_len").value(), p.getScalar<int32_t>("max_beam_width").value(),
            static_cast<bool>(p.getScalar<int32_t>("paged_kv_cache").value()));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* IBGPTAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GPTAttentionPlugin::destroy()
    try
    {
        auto* obj = new IBGPTAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void IBGPTAttentionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* IBGPTAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
