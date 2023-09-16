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
#include "tensorrt_llm/plugins/rmsnormPlugin/rmsnormPlugin.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::RmsNormPluginCreator;
using nvinfer1::plugin::RmsNormPlugin;

static const char* RMSNORM_PLUGIN_VERSION{"1"};
static const char* RMSNORM_PLUGIN_NAME{"RmsNorm"};
PluginFieldCollection RmsNormPluginCreator::mFC{};
std::vector<PluginField> RmsNormPluginCreator::mPluginAttributes;

RmsNormPlugin::RmsNormPlugin(float eps, nvinfer1::DataType type)
    : mEps(eps)
    , mType(type)
{
}

// Parameterized constructor
RmsNormPlugin::RmsNormPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mEps);
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RmsNormPlugin::clone() const noexcept
{
    auto* plugin = new RmsNormPlugin(mEps, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RmsNormPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[outputIndex];
}

bool RmsNormPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(0 <= pos && pos < 5);
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void RmsNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RmsNormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RmsNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    //     bias nullptr
    // outputs
    //     output [M(*), N]

    int m = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[0].dims.d[i];
    }
    const int n = inputDesc[1].dims.d[0];

    if (mType == DataType::kHALF)
    {
        const half* input = reinterpret_cast<const half*>(inputs[0]);
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        // No bias in RmsNorm
        // const half* bias = reinterpret_cast<const half*>(inputs[2]);
        half* bias = nullptr;
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, bias, mEps, m, n, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* input = reinterpret_cast<const float*>(inputs[0]);
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        // No bias in RmsNorm
        // const float* bias = reinterpret_cast<const float*>(inputs[2]);
        float* bias = nullptr;
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, bias, mEps, m, n, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        const __nv_bfloat16* input = reinterpret_cast<const __nv_bfloat16*>(inputs[0]);
        const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
        // No bias in RmsNorm
        // const __nv_bfloat16* bias = reinterpret_cast<const __nv_bfloat16*>(inputs[2]);
        __nv_bfloat16* bias = nullptr;
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeGeneralRmsNorm(output, input, weight, bias, mEps, m, n, stream);
    }
#endif

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RmsNormPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* RmsNormPlugin::getPluginType() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsNormPlugin::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

int RmsNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RmsNormPlugin::initialize() noexcept
{
    return 0;
}

void RmsNormPlugin::terminate() noexcept {}

size_t RmsNormPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mType);
}

void RmsNormPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void RmsNormPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void RmsNormPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RmsNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

RmsNormPluginCreator::RmsNormPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1e-5f));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RmsNormPluginCreator::getPluginName() const noexcept
{
    return RMSNORM_PLUGIN_NAME;
}

const char* RmsNormPluginCreator::getPluginVersion() const noexcept
{
    return RMSNORM_PLUGIN_VERSION;
}

const PluginFieldCollection* RmsNormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RmsNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    float eps;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RmsNormPlugin(eps, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RmsNormPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RmsNormPlugin::destroy()
    try
    {
        auto* obj = new RmsNormPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void RmsNormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* RmsNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
