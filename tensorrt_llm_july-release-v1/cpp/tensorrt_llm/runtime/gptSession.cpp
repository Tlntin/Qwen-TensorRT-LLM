//
// Created by martinma on 5/24/23.
//
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

#include "tensorrt_llm/runtime/gptSession.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <fstream>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

namespace
{

nvinfer1::DataType getTensorDataType(nvinfer1::ICudaEngine const& engine, std::string const& name)
{
    return engine.getTensorDataType(name.c_str());
}

} // namespace

GptSession::GptSession(GptModelConfig const& modelConfig, WorldConfig const& worldConfig, void const* engineBuffer,
    std::size_t engineSize, LoggerPtr logger)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{initDevice(worldConfig)}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_shared<TllmRuntime>(engineBuffer, engineSize, *mLogger)}
    , mDecoder{}
{
    TLLM_CHECK_WITH_INFO(mRuntime->getNbProfiles() == 1, "GPT only expects one optimization profile");
    createContexts();
    mBuffers.create(*mRuntime, mModelConfig);
    // TODO compare expected and runtime tensor names?
}

void GptSession::createContexts()
{
    mRuntime->clearContexts();
    // Instantiate two contexts for flip-flopping
    mRuntime->addContext(0);
    mRuntime->addContext(0);
}

namespace
{
std::vector<ITensor::SharedPtr> createBufferVector(
    TllmRuntime const& runtime, SizeType const numBuffers, std::string const& prefix)
{
    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    std::vector<ITensor::SharedPtr> vector;

    for (SizeType i = 0; i < numBuffers; ++i)
    {
        std::string name{prefix + std::to_string(i)};
        auto type = getTensorDataType(engine, name);
        vector.emplace_back(manager.emptyTensor(MemoryType::kGPU, type));
    }
    return vector;
}

void reshapeBufferVector(std::vector<ITensor::SharedPtr>& vector, nvinfer1::Dims const& shape)
{
    for (auto& buffer : vector)
    {
        buffer->reshape(shape);
    }
}
} // namespace

void GptSession::RuntimeBuffers::create(TllmRuntime& runtime, GptModelConfig const& modelConfig)
{
    auto& manager = runtime.getBufferManager();

    auto logitsType = getTensorDataType(runtime.getEngine(), "logits");
    logits = manager.emptyTensor(MemoryType::kGPU, logitsType);

    presentKeysVals = createBufferVector(runtime, modelConfig.getNbLayers(), "present_key_value_");

    if (modelConfig.useGptAttentionPlugin() || modelConfig.useInflightBatchingGptAttentionPlugin())
    {
        sequenceLengths = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        if (modelConfig.useGptAttentionPlugin())
        {
            // allocate here because it never changes size
            pastKeyValueLengths = manager.cpu(ITensor::makeShape({2}), nvinfer1::DataType::kINT32);
        }
        else
        {
            pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        }
    }
    else
    {
        presentKeysValsAlt = createBufferVector(runtime, modelConfig.getNbLayers(), "present_key_value_");
    }

    if (modelConfig.useInflightBatchingGptAttentionPlugin())
    {
        beamWidths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        for (SizeType i = 0; i < modelConfig.getNbLayers(); ++i)
        {
            keyValuePointers.emplace_back(manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32));
        }
        requestCacheMaxSeqLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        hostInputLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        inputLengths2 = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        cacheIndirectionPointers = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }

    maxInputLength = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirection0 = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    cacheIndirection1 = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    cumLogProbs = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    parentIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    inputOffsets = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
}

void GptSession::RuntimeBuffers::reshape(
    GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, SizeType worldSize)
{
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxSeqLength = generationConfig.getMaxSeqLength();

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldSize);
    // logits are tiled to {batchSize, beamWidth, vocabSizePadded} after context step of engine
    logits->reshape(ITensor::makeShape({batchSize, 1, vocabSizePadded}));

    auto const cacheShape = ITensor::makeShape(
        {batchSize, 2, modelConfig.getNbHeads(), maxSeqLength, modelConfig.getHiddenSize() / modelConfig.getNbHeads()});

    reshapeBufferVector(presentKeysVals, cacheShape);

    if (modelConfig.useGptAttentionPlugin() || modelConfig.useInflightBatchingGptAttentionPlugin())
    {
        // reserve {batchSize * beamWidth}, but reduce to {batchSize} for context step
        sequenceLengths->reshape(ITensor::makeShape({batchSize * beamWidth}));
        sequenceLengths->reshape(ITensor::makeShape({batchSize}));
    }
    else
    {
        reshapeBufferVector(presentKeysValsAlt, cacheShape);
    }

    if (modelConfig.useInflightBatchingGptAttentionPlugin())
    {
        beamWidths->reshape(ITensor::makeShape({batchSize}));
        requestCacheMaxSeqLengths->reshape(ITensor::makeShape({batchSize}));
        hostInputLengths->reshape(ITensor::makeShape({batchSize}));

        pastKeyValueLengths->reshape(ITensor::makeShape({batchSize}));

        // We store int64 values in int32 tensors. That's why multiplier 2
        for (SizeType i = 0; i < modelConfig.getNbLayers(); ++i)
        {
            keyValuePointers[i]->reshape(ITensor::makeShape({batchSize, 2}));
        }
        cacheIndirectionPointers->reshape(ITensor::makeShape({batchSize, 2}));
    }

    maxInputLength->reshape(ITensor::makeShape({generationConfig.maxInputLength}));
    cacheIndirection0->reshape(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));
    cacheIndirection1->reshape(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}));

    cumLogProbs->reshape(ITensor::makeShape({batchSize * beamWidth}));

    allocated = true;
}

void GptSession::RuntimeBuffers::tile(BufferManager& manager, GenerationConfig const& generationConfig, bool usePlugin)
{
    auto& stream = manager.getStream();
    auto const beamWidth = generationConfig.beamWidth;

    auto tileBuffer = [&manager, &stream, beamWidth](TensorPtr& tensor)
    {
        if (tensor)
        {
            auto shape = tensor->getShape();
            shape.d[0] *= beamWidth;
            auto tiledTensor = std::shared_ptr(manager.gpu(shape, tensor->getDataType()));
            switch (tensor->getDataType())
            {
            case nvinfer1::DataType::kINT32:
                kernels::invokeTileTensor<SizeType>(*tiledTensor, *tensor, beamWidth, stream);
                break;
            case nvinfer1::DataType::kFLOAT:
                kernels::invokeTileTensor<float>(*tiledTensor, *tensor, beamWidth, stream);
                break;
            case nvinfer1::DataType::kHALF:
                kernels::invokeTileTensor<half>(*tiledTensor, *tensor, beamWidth, stream);
                break;
            default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
            }
            stream.synchronize();
            tensor = tiledTensor;
        }
    };

    // logits needs beamWidth in second dimension
    auto logitsShape = logits->getShape();
    logitsShape.d[1] *= beamWidth;
    tileBuffer(logits);
    logits->reshape(logitsShape);

    tileBuffer(attentionMask);
    tileBuffer(tokenMask);
    tileBuffer(inputLengths);

    // no need to copy data in sequenceLengths because it is overwritten in prepareNextStep
    auto const batchSize = generationConfig.batchSize;
    if (usePlugin)
        sequenceLengths->reshape(ITensor::makeShape({batchSize * beamWidth}));

    // no need to copy data in lastTokenIds because it is overwritten in prepareNextStep
    lastTokenIds->reshape(ITensor::makeShape({batchSize * beamWidth}));

    for (auto& buffer : presentKeysVals)
        tileBuffer(buffer);
    for (auto& buffer : presentKeysValsAlt)
        tileBuffer(buffer);
}

void GptSession::setup(SizeType const batchSize, SizeType const maxInputLength, SizeType const maxNewTokens,
    SamplingConfig const& samplingConfig)
{
    // Store these params related to buffer size to check against
    // the input shape with the params given in generate()
    mGenerationConfig = {batchSize, maxInputLength, maxNewTokens, samplingConfig.beamWidth};

    mDecoder = createDecoder();
    mDecoder->setup(samplingConfig, batchSize);

    mBuffers.reshape(mGenerationConfig, mModelConfig, mWorldConfig.getSize());
}

std::unique_ptr<IGptDecoder> GptSession::createDecoder()
{
    return IGptDecoder::create(nvinfer1::DataType::kFLOAT, mModelConfig.getVocabSize(),
        mModelConfig.getVocabSizePadded(mWorldConfig.getSize()), mRuntime->getStreamPtr());
}

namespace
{
void insertTensorVector(StringPtrMap<ITensor>& map, std::string const& key, std::vector<ITensor::SharedPtr> const& vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        map.insert_or_assign(key + std::to_string(i), vec[i]);
}
} // namespace

void GptSession::getContextBuffers(
    TensorMap& inputBuffers, TensorMap& outputBuffers, TensorPtr const& inputIds, TensorPtr const& inputLengths) const
{
    inputBuffers.clear();
    outputBuffers.clear();

    inputBuffers.insert_or_assign("input_ids", inputIds);
    inputBuffers.insert_or_assign("max_input_length", mBuffers.maxInputLength);
    inputBuffers.insert_or_assign("position_ids", mBuffers.positionIds);
    inputBuffers.insert_or_assign("last_token_ids", mBuffers.lastTokenIds);

    outputBuffers.insert_or_assign("logits", ITensor::view(mBuffers.logits)); // feed a view to TensorRT runtime

    if (mModelConfig.useGptAttentionPlugin() || mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        insertTensorVector(inputBuffers, "past_key_value_", mBuffers.presentKeysVals);
        inputBuffers.insert_or_assign("past_key_value_length", mBuffers.pastKeyValueLengths);
    }
    else
    {
        auto& engine = mRuntime->getEngine();
        auto& manager = mRuntime->getBufferManager();
        auto const kvCacheShape = ITensor::makeShape({mGenerationConfig.batchSize, 2, mModelConfig.getNbHeads(), 0,
            mModelConfig.getHiddenSize() / mModelConfig.getNbHeads()});

        for (SizeType i = 0; i < mModelConfig.getNbLayers(); ++i)
        {
            std::string name = "past_key_value_" + std::to_string(i);
            TensorPtr tmp = manager.gpu(kvCacheShape, getTensorDataType(engine, name));
            inputBuffers.insert_or_assign(name, std::move(tmp));
        }
    }

    if (mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("input_lengths", mBuffers.inputLengths2);
        inputBuffers.insert_or_assign("beam_widths", mBuffers.beamWidths);
        inputBuffers.insert_or_assign("host_input_lengths", mBuffers.hostInputLengths);
        inputBuffers.insert_or_assign("cache_indir_pointers", mBuffers.cacheIndirectionPointers);
        insertTensorVector(inputBuffers, "past_key_value_pointers_", mBuffers.keyValuePointers);
        inputBuffers.insert_or_assign("req_cache_max_seq_lengths", mBuffers.requestCacheMaxSeqLengths);
    }
    else
    {
        inputBuffers.insert_or_assign("input_lengths", mBuffers.inputLengths);
        inputBuffers.insert_or_assign("cache_indirection", mBuffers.cacheIndirection0);
    }

    insertTensorVector(outputBuffers, "present_key_value_", mBuffers.presentKeysVals);

    if (mModelConfig.useGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("sequence_length", mBuffers.sequenceLengths);
        inputBuffers.insert_or_assign("masked_tokens", mBuffers.tokenMask);
    }
    else if (!mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("attention_mask", mBuffers.attentionMask);
    }
}

void GptSession::getNextStepBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType const step,
    TensorPtr const& inputIds, TensorPtr const& inputLengths) const
{
    inputBuffers.clear();
    outputBuffers.clear();

    inputBuffers.insert_or_assign("input_ids", inputIds);
    inputBuffers.insert_or_assign("max_input_length", mBuffers.maxInputLength);
    inputBuffers.insert_or_assign("position_ids", mBuffers.positionIds);
    inputBuffers.insert_or_assign("last_token_ids", mBuffers.lastTokenIds);

    outputBuffers.insert_or_assign("logits", ITensor::view(mBuffers.logits)); // feed a view to TensorRT runtime

    if (mModelConfig.useGptAttentionPlugin() || mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        insertTensorVector(inputBuffers, "past_key_value_", mBuffers.presentKeysVals);
        insertTensorVector(outputBuffers, "present_key_value_", mBuffers.presentKeysVals);

        inputBuffers.insert_or_assign("past_key_value_length", mBuffers.pastKeyValueLengths);
    }
    else
    {
        insertTensorVector(
            inputBuffers, "past_key_value_", (step % 2) ? mBuffers.presentKeysValsAlt : mBuffers.presentKeysVals);
        insertTensorVector(
            outputBuffers, "present_key_value_", (step % 2) ? mBuffers.presentKeysVals : mBuffers.presentKeysValsAlt);
    }

    if (mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("input_lengths", mBuffers.inputLengths2);
        inputBuffers.insert_or_assign("beam_widths", mBuffers.beamWidths);
        inputBuffers.insert_or_assign("host_input_lengths", mBuffers.hostInputLengths);
        inputBuffers.insert_or_assign("cache_indir_pointers", mBuffers.cacheIndirectionPointers);
        insertTensorVector(inputBuffers, "past_key_value_pointers_", mBuffers.keyValuePointers);
        inputBuffers.insert_or_assign("req_cache_max_seq_lengths", mBuffers.requestCacheMaxSeqLengths);
    }
    else
    {
        inputBuffers.insert_or_assign("input_lengths", mBuffers.inputLengths);
        inputBuffers.insert_or_assign(
            "cache_indirection", (step % 2) ? mBuffers.cacheIndirection0 : mBuffers.cacheIndirection1);
    }

    if (mModelConfig.useGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("sequence_length", mBuffers.sequenceLengths);
        inputBuffers.insert_or_assign("masked_tokens", mBuffers.tokenMask);
    }
    else if (!mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("attention_mask", mBuffers.attentionMask);
    }
}

void GptSession::setRawPointers(
    TensorPtr& pointers, TensorPtr const& input, int32_t leadingDimMultiplier, const nvinfer1::DataType& type)
{
    switch (type)
    {
    case nvinfer1::DataType::kHALF: setRawPointers_<half>(pointers, input, leadingDimMultiplier); break;
    case nvinfer1::DataType::kFLOAT: setRawPointers_<float>(pointers, input, leadingDimMultiplier); break;
    case nvinfer1::DataType::kINT32: setRawPointers_<int32_t>(pointers, input, leadingDimMultiplier); break;
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

void GptSession::prepareContextStep(GenerationInput const& inputs)
{
    SizeType const step{0};

    auto& manager = mRuntime->getBufferManager();
    auto& stream = mRuntime->getStream();

    if (mModelConfig.useGptAttentionPlugin() || mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        auto pastKeyValueLengths = bufferCast<SizeType>(*mBuffers.pastKeyValueLengths);
        if (mModelConfig.useGptAttentionPlugin())
        {
            pastKeyValueLengths[0] = 0; // past_key_value_length
            pastKeyValueLengths[1] = 1; // is_context
        }
        else
        {
            for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
            {
                pastKeyValueLengths[i] = 0;
            }
        }

        if (inputs.packed)
        {
            auto const inputOffsets = manager.copyFrom(*mBuffers.inputOffsets, MemoryType::kCPU);
            auto const* inputOffsetsData = bufferCast<SizeType>(*inputOffsets);

            std::vector<SizeType> positionIdsVec(inputs.ids->getShape().d[1]);
            for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
                std::iota(std::begin(positionIdsVec) + inputOffsetsData[i],
                    std::begin(positionIdsVec) + inputOffsetsData[i + 1], 0);
            mBuffers.positionIds = manager.copyFrom(positionIdsVec, inputs.ids->getShape(), MemoryType::kGPU);
        }
        else
        {
            std::vector<SizeType> positionIdsVec(inputs.ids->getSize());
            for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
                std::iota(std::begin(positionIdsVec) + i * mGenerationConfig.maxInputLength,
                    std::begin(positionIdsVec) + (i + 1) * mGenerationConfig.maxInputLength, 0);
            mBuffers.positionIds = manager.copyFrom(positionIdsVec, inputs.ids->getShape(), MemoryType::kGPU);
        }
    }
    else
    {
        mBuffers.attentionMask = manager.copyFrom(*inputs.ids, MemoryType::kGPU);
        kernels::invokeBuildAttentionMask(*mBuffers.attentionMask, inputs.padId, stream);

        auto attentionMaskHost = manager.copyFrom(*mBuffers.attentionMask, MemoryType::kCPU);
        auto const* attentionMaskData = reinterpret_cast<SizeType const*>(attentionMaskHost->data());
        std::vector<SizeType> positionIdsVec(mBuffers.attentionMask->getSize());
        for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
        {
            std::exclusive_scan(attentionMaskData + i * mGenerationConfig.maxInputLength,
                attentionMaskData + (i + 1) * mGenerationConfig.maxInputLength,
                std::begin(positionIdsVec) + i * mGenerationConfig.maxInputLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskData[i] == 0)
                positionIdsVec[i] = 1;
        mBuffers.positionIds = manager.copyFrom(positionIdsVec, mBuffers.attentionMask->getShape(), MemoryType::kGPU);
    }

    if (mModelConfig.useGptAttentionPlugin())
    {
        mBuffers.tokenMask
            = manager.gpu(ITensor::makeShape({mGenerationConfig.batchSize, mGenerationConfig.getMaxSeqLength()}),
                nvinfer1::DataType::kINT32);
        kernels::invokeBuildTokenMask(*mBuffers.tokenMask, *inputs.lengths, mGenerationConfig.maxInputLength, stream);

        kernels::invokeFill(*mBuffers.sequenceLengths, mGenerationConfig.maxInputLength + step, stream);
    }

    if (mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        auto beamWidths = bufferCast<SizeType>(*mBuffers.beamWidths);
        auto requestCacheMaxSeqLengths = bufferCast<SizeType>(*mBuffers.requestCacheMaxSeqLengths);
        for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
        {
            beamWidths[i] = 1;
            requestCacheMaxSeqLengths[i] = mGenerationConfig.getMaxSeqLength();
        }
        mBuffers.hostInputLengths = manager.copyFrom(*mBuffers.inputLengths, MemoryType::kCPU);

        mBuffers.inputLengths2->reshape(mBuffers.inputLengths->getShape());
        manager.copy(*mBuffers.inputLengths, *mBuffers.inputLengths2);

        auto kvType = getTensorDataType(mRuntime->getEngine(), "past_key_value_0");
        for (SizeType i = 0; i < mModelConfig.getNbLayers(); ++i)
        {
            setRawPointers(mBuffers.keyValuePointers[i], mBuffers.presentKeysVals[i], 1, kvType);
        }
        setRawPointers(mBuffers.cacheIndirectionPointers, mBuffers.cacheIndirection0, 1, nvinfer1::DataType::kINT32);
    }

    if (inputs.packed)
    {
        mBuffers.lastTokenIds = manager.copyFrom(*ITensor::slice(mBuffers.inputOffsets, 1), MemoryType::kGPU);
    }
    else
    {
        mBuffers.lastTokenIds = manager.copyFrom(*inputs.lengths, MemoryType::kGPU);
    }

    manager.setZero(*mBuffers.cacheIndirection0);
    manager.setZero(*mBuffers.cacheIndirection1);
};

GptSession::TensorPtr GptSession::prepareNextStep(
    SizeType const step, bool const inputPacked, TensorPtr const& outputIds)
{
    auto& manager = mRuntime->getBufferManager();
    auto& stream = mRuntime->getStream();

    // Needs to be reshuffled with every new request for in-flight batching
    TensorPtr nextInputIds = ITensor::slice(outputIds, mGenerationConfig.maxInputLength + step, 1);
    if (inputPacked)
    {
        // squeeze first dim and batch in last dim
        nextInputIds->reshape(ITensor::makeShape({1, mGenerationConfig.batchSize * mGenerationConfig.beamWidth}));
    }
    else
    {
        // squeeze first dim
        nextInputIds->reshape(ITensor::makeShape({mGenerationConfig.batchSize * mGenerationConfig.beamWidth, 1}));
    }

    if (mModelConfig.useGptAttentionPlugin() || mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        auto pastKeyValueLengths = bufferCast<SizeType>(*mBuffers.pastKeyValueLengths);
        if (mModelConfig.useGptAttentionPlugin())
        {
            kernels::invokeFill(*mBuffers.sequenceLengths, mGenerationConfig.maxInputLength + step, stream);

            pastKeyValueLengths[0] = mGenerationConfig.maxInputLength + step; // past_key_value_length
            pastKeyValueLengths[1] = 0;                                       // is_context
        }
        else
        {
            // Fill numReq input lengths for each request. E.g. if req 0 is in generation mode generating nth token
            // And req 1 is generating mth token
            // mBuffers.pastKeyValueLengths should contain [n, m]
            // FIXME(nkorobov): we don't need to copy this array every timestep
            auto inputHostLengthCPU = manager.copyFrom(*mBuffers.inputLengths, MemoryType::kCPU);
            auto inputHostLengthCPUPtr = bufferCast<SizeType>(*inputHostLengthCPU);
            for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
            {
                pastKeyValueLengths[i] = inputHostLengthCPUPtr[i * mGenerationConfig.beamWidth] + step;
            }
        }

        mBuffers.positionIds->reshape(mBuffers.inputLengths->getShape());
        manager.copy(*mBuffers.inputLengths, *mBuffers.positionIds);
        kernels::invokeAdd(*mBuffers.positionIds, step, stream);

        auto const size = static_cast<SizeType>(mBuffers.positionIds->getSize());
        if (inputPacked)
            mBuffers.positionIds->reshape(ITensor::makeShape({1, size}));
        else
            mBuffers.positionIds->reshape(ITensor::makeShape({size, 1}));
    }
    else
    {
        auto attentionMaskHost = manager.copyFrom(*mBuffers.attentionMask, MemoryType::kCPU);
        auto const* attentionMaskData = bufferCast<SizeType>(*attentionMaskHost);
        auto const shape = mBuffers.attentionMask->getShape();
        auto const nbInputs = shape.d[0];
        auto const oldLength = shape.d[1];
        auto const newLength = oldLength + 1;
        auto const newShape = ITensor::makeShape({nbInputs, newLength});
        std::vector<SizeType> attentionMaskVec(ITensor::volume(newShape));
        for (SizeType i = 0; i < nbInputs; ++i)
        {
            std::copy(attentionMaskData + i * oldLength, attentionMaskData + (i + 1) * oldLength,
                std::begin(attentionMaskVec) + i * newLength);
            attentionMaskVec[(i + 1) * newLength - 1] = 1;
        }
        mBuffers.attentionMask = manager.copyFrom(attentionMaskVec, newShape, MemoryType::kGPU);

        // TODO old positionIds could be recovered to avoid scan
        std::vector<SizeType> positionIdsVec(attentionMaskVec.size());
        for (SizeType i = 0; i < nbInputs; ++i)
        {
            std::exclusive_scan(attentionMaskVec.begin() + i * newLength,
                attentionMaskVec.begin() + (i + 1) * newLength, std::begin(positionIdsVec) + i * newLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskVec[i] == 0)
                positionIdsVec[i] = 1;
        std::vector<SizeType> positionIdsEndVec(nbInputs);
        for (SizeType i = 0; i < nbInputs; ++i)
            positionIdsEndVec[i] = positionIdsVec[(i + 1) * newLength - 1];

        mBuffers.positionIds = manager.copyFrom(positionIdsEndVec, ITensor::makeShape({nbInputs, 1}), MemoryType::kGPU);
    }

    if (mModelConfig.useInflightBatchingGptAttentionPlugin())
    {
        auto beamWidths = bufferCast<SizeType>(*mBuffers.beamWidths);
        auto requestCacheMaxSeqLengths = bufferCast<SizeType>(*mBuffers.requestCacheMaxSeqLengths);
        // Fill beam width for each request. E.g. if req 0 is in generation mode and req 1 is in context mode
        // mBuffers.beamWidths should contain [beam_width_req_0, 1]
        for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
        {
            beamWidths[i] = mGenerationConfig.beamWidth;
            requestCacheMaxSeqLengths[i] = mGenerationConfig.getMaxSeqLength();
        }

        // Fill numReq input lengths for each request. E.g. if req 0 is in generation mode and req 1 is in context mode
        // mBuffers.hostInputLengths should contain [1, in_seq_len_req_1]
        auto hostInputLengths = bufferCast<SizeType>(*mBuffers.hostInputLengths);
        for (SizeType i = 0; i < mGenerationConfig.batchSize; ++i)
        {
            hostInputLengths[i] = 1;
        }

        // Fill numSeq input lengths for each sequence. E.g. if req 0 is in generation mode and req 1 is in context mode
        // and beam_width = 2, mBuffers.inputLengths2 should contain [1, 1, in_seq_len_req_1]
        const auto inputLengthsCPU = manager.cpu(mBuffers.inputLengths->getShape(), nvinfer1::DataType::kINT32);
        auto inputLengthsCPUPtr = bufferCast<SizeType>(*inputLengthsCPU);
        for (SizeType i = 0; i < mBuffers.inputLengths->getShape().d[0]; ++i)
        {
            inputLengthsCPUPtr[i] = 1;
        }
        mBuffers.inputLengths2 = manager.copyFrom(*inputLengthsCPU, MemoryType::kGPU);

        // Setup pointers to the requests for indirect cache and key value
        auto kvType = getTensorDataType(mRuntime->getEngine(), "past_key_value_0");
        for (SizeType i = 0; i < mModelConfig.getNbLayers(); ++i)
        {
            setRawPointers(
                mBuffers.keyValuePointers[i], mBuffers.presentKeysVals[i], mGenerationConfig.beamWidth, kvType);
        }
        setRawPointers(mBuffers.cacheIndirectionPointers,
            (step % 2) ? mBuffers.cacheIndirection0 : mBuffers.cacheIndirection1, 1, nvinfer1::DataType::kINT32);
    }

    kernels::invokeFill(*mBuffers.lastTokenIds, 1, stream);
    if (inputPacked)
    {
        kernels::invokeInclusiveSum(*mBuffers.lastTokenIds, *mBuffers.lastTokenIds, manager, stream);
    }

    return nextInputIds;
};

void GptSession::copyInputToOutput(TensorPtr const& outputIds, TensorPtr const& inputIds, TensorPtr const& inputLengths,
    SizeType const padId, SizeType const maxInputLength, bool const inputPacked) const
{
    auto& stream = mRuntime->getStream();

    if (inputPacked)
    {
        kernels::invokeCopyPackedInputToOutput(
            *outputIds, *inputIds, *mBuffers.inputOffsets, maxInputLength, padId, stream);
    }
    else
    {
        kernels::invokeCopyInputToOutput(*outputIds, *inputIds, *inputLengths, padId, stream);
    }
}

void GptSession::checkInputShape(TensorPtr const& inputIds, TensorPtr const& inputLengths, bool const inputPacked) const
{
    auto const batchSize = mGenerationConfig.batchSize;
    auto const maxInputLength = mGenerationConfig.maxInputLength;

    BufferManager& manager = mRuntime->getBufferManager();

    TLLM_CHECK_WITH_INFO(inputLengths->getShape().d[0] == batchSize,
        "Given batch size is different from the one used in setup(), "
        "rerun the setup function with the new batch size to avoid buffer overflow.");
    auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto const* inputLengthsData = bufferCast<SizeType>(*inputLengthsHost);
    auto const inputLengthMax = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());
    TLLM_CHECK_WITH_INFO(inputLengthMax <= maxInputLength,
        "Given input length is larger than the max used in setup(), "
        "rerun the setup function with the new max input length to avoid buffer overflow.");

    if (inputPacked)
    {
        auto const inputLengthSum = std::reduce(inputLengthsData, inputLengthsData + inputLengths->getSize());
        TLLM_CHECK_WITH_INFO(inputIds->getShape().d[0] == 1 && inputIds->getShape().d[1] == inputLengthSum,
            "Packed input must have shape [1, <sum of input lengths>].");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(inputIds->getShape().d[0] == batchSize && inputIds->getShape().d[1] == maxInputLength,
            "Input must have shape [batch size, max input length]");
    }
}

void GptSession::generate(GenerationOutput& outputs, GenerationInput const& inputs)
{
    // parameters from previous call to setup
    auto const batchSize = mGenerationConfig.batchSize;
    auto const beamWidth = mGenerationConfig.beamWidth;
    auto const maxInputLength = mGenerationConfig.maxInputLength;
    auto const maxNewTokens = mGenerationConfig.maxNewTokens;
    auto const maxSeqLength = mGenerationConfig.getMaxSeqLength();
    auto finalSeqLength = maxSeqLength;

    TLLM_CHECK_WITH_INFO(inputs.packed == mModelConfig.isInputPacked(),
        "The chosen model requires a packed input tensor (did you set packed?).");

    checkInputShape(inputs.ids, inputs.lengths, inputs.packed);
    mBuffers.reshape(mGenerationConfig, mModelConfig, mWorldConfig.getSize());

    TLLM_CHECK_WITH_INFO(mBuffers.allocated, "Buffers not allocated, please call setup first!");

    auto& manager = mRuntime->getBufferManager();
    auto& stream = mRuntime->getStream();

    mBuffers.inputLengths = inputs.lengths;
    if (inputs.packed)
    {
        mBuffers.inputOffsets->reshape(ITensor::makeShape({batchSize + 1}));
        manager.setZero(*mBuffers.inputOffsets);
        kernels::invokeInclusiveSum(*ITensor::slice(mBuffers.inputOffsets, 1), *inputs.lengths, manager, stream);
    }

    // input for decoding
    auto const batchSizeDims = ITensor::makeShape({batchSize});
    auto const batchSizeXbeamWidthDims = ITensor::makeShape({batchSize, beamWidth});
    std::vector<std::int32_t> endIdsVec(batchSize * beamWidth, inputs.endId);
    auto endIds = std::shared_ptr(manager.copyFrom(endIdsVec, batchSizeXbeamWidthDims, MemoryType::kGPU));
    DecodingInput decodingInput{0, maxInputLength, batchSize, mBuffers.logits, endIds};
    std::vector<std::int32_t> sequenceLimitLengthsVec(batchSize, maxSeqLength);
    decodingInput.sequenceLimitLength = manager.copyFrom(sequenceLimitLengthsVec, batchSizeDims, MemoryType::kGPU);
    decodingInput.embeddingBias = inputs.embeddingBiasOpt;
    decodingInput.lengths = inputs.lengths;
    decodingInput.badWordsList = inputs.badWordsList;
    decodingInput.stopWordsList = inputs.stopWordsList;

    // output for decoding
    auto const outputIdsShape = ITensor::makeShape({maxSeqLength, batchSize, beamWidth});
    outputs.ids->reshape(outputIdsShape);
    if (inputs.disableInputCopy)
    {
        manager.setZero(*outputs.ids);
    }
    else
    {
        copyInputToOutput(outputs.ids, inputs.ids, inputs.lengths, inputs.padId, maxInputLength, inputs.packed);
    }

    DecodingOutput decodingOutput{outputs.ids};
    auto finished = std::shared_ptr(manager.gpu(batchSizeXbeamWidthDims, nvinfer1::DataType::kBOOL));
    manager.setZero(*finished);
    decodingOutput.finished = finished;
    std::vector<std::int32_t> sequenceLengthsVec(batchSize * beamWidth, maxInputLength);
    decodingOutput.lengths = manager.copyFrom(sequenceLengthsVec, batchSizeXbeamWidthDims, MemoryType::kGPU);
    if (beamWidth > 1)
    {
        mBuffers.parentIds->reshape(outputIdsShape);
        manager.setZero(*mBuffers.parentIds);
        decodingOutput.parentIds = mBuffers.parentIds;
    }
    std::vector<float> cumLogProbsHost(batchSize * beamWidth, -1e20f);
    for (SizeType i = 0; i < batchSize; ++i)
        cumLogProbsHost[i * beamWidth] = 0;
    manager.copy(cumLogProbsHost.data(), *mBuffers.cumLogProbs);
    decodingOutput.cumLogProbs = mBuffers.cumLogProbs;
    if (outputs.logProbs)
    {
        outputs.logProbs->reshape(ITensor::makeShape({maxNewTokens, batchSize * beamWidth}));
        manager.setZero(*outputs.logProbs);
        decodingOutput.logProbs = outputs.logProbs;
    }

    TensorMap inputBuffers[2];
    TensorMap outputBuffers[2];
    auto& onTokenGenerated = outputs.onTokenGenerated;

    for (SizeType step = 0; step < maxNewTokens; ++step)
    {
        auto const contextId = step % 2;
        if (step == 0)
        {
            prepareContextStep(inputs);
            getContextBuffers(inputBuffers[contextId], outputBuffers[contextId], inputs.ids, inputs.lengths);
            mRuntime->setInputTensors(contextId, inputBuffers[contextId]);
            mRuntime->setOutputTensors(contextId, outputBuffers[contextId]);
        }

        bool enqueueSuccessful = mRuntime->executeContext(contextId);
        TLLM_CHECK_WITH_INFO(enqueueSuccessful, "Executing TRT engine failed!");

        if (step == 0 && beamWidth > 1)
        {
            mBuffers.tile(mRuntime->getBufferManager(), mGenerationConfig, mModelConfig.useGptAttentionPlugin());
            decodingInput.logits = mBuffers.logits;
            decodingInput.lengths = mBuffers.inputLengths;
        }

        if (step < maxNewTokens - 1)
        {
            auto const nextContextId = (step + 1) % 2;
            auto nextInputIds = prepareNextStep(step, inputs.packed, outputs.ids);
            getNextStepBuffers(
                inputBuffers[nextContextId], outputBuffers[nextContextId], step, nextInputIds, mBuffers.inputLengths);
            mRuntime->setInputTensors(nextContextId, inputBuffers[nextContextId]);
            mRuntime->setOutputTensors(nextContextId, outputBuffers[nextContextId]);
        }

        if (step % 2)
        {
            decodingInput.cacheIndirection = mBuffers.cacheIndirection1;
            decodingOutput.cacheIndirection = mBuffers.cacheIndirection0;
        }
        else
        {
            decodingInput.cacheIndirection = mBuffers.cacheIndirection0;
            decodingOutput.cacheIndirection = mBuffers.cacheIndirection1;
        }
        decodingInput.step = step + maxInputLength;

        auto const shouldStop = mDecoder->forward(decodingOutput, decodingInput);

        if (onTokenGenerated)
        {
            onTokenGenerated(outputs.ids, step, shouldStop || step == maxNewTokens - 1);
        }

        if (shouldStop)
        {
            finalSeqLength = maxInputLength + step;
            outputs.ids->reshape(ITensor::makeShape({finalSeqLength, batchSize, beamWidth}));
            mLogger->log(nvinfer1::ILogger::Severity::kVERBOSE, "GPT decoding finished early");
            break;
        }
    }
    gatherTree(outputs.ids, decodingOutput.lengths, endIds, finalSeqLength);
}

// this should be similar to gatherTree in cpp/tensorrt_llm/thop/gatherTreeOp.cpp
void GptSession::gatherTree(
    TensorPtr& outputIds, TensorPtr const& sequenceLength, TensorPtr const& endIds, SizeType const finalSeqLength) const
{
    auto const batchSize = mGenerationConfig.batchSize;
    auto const beamWidth = mGenerationConfig.beamWidth;
    auto const maxInputLength = mGenerationConfig.maxInputLength;

    auto const& manager = mRuntime->getBufferManager();
    auto const& stream = mRuntime->getStream();

    auto workspace = manager.gpu(batchSize * beamWidth * finalSeqLength, nvinfer1::DataType::kINT32);
    manager.setZero(*workspace);
    auto tmpOutputIds = manager.copyFrom(*outputIds, MemoryType::kGPU);

    // For sampling, it is equivalent to all parent ids are 0.
    tensorrt_llm::kernels::gatherTreeParam param;
    param.beams = bufferCast<SizeType>(*workspace);
    // Remove prompt length if possible
    param.max_sequence_lengths = bufferCast<SizeType>(*sequenceLength);
    // add sequence_length 1 here because the sequence_length of time step t is t - 1
    param.max_sequence_length_final_step = 1;
    // response input lengths (used to slice the ids during postprocessing), used in interactive generation
    // This feature is not supported yet, setting it to nullptr temporarily.
    param.response_input_lengths = nullptr;
    param.max_time = finalSeqLength;
    param.batch_size = batchSize;
    param.beam_width = beamWidth;
    param.step_ids = bufferCast<SizeType>(*tmpOutputIds);
    param.parent_ids = bufferCast<SizeType>(*mBuffers.parentIds);
    param.end_tokens = bufferCast<SizeType>(*endIds);
    param.max_input_length = maxInputLength;
    param.input_lengths = bufferCast<SizeType>(*mBuffers.inputLengths);

    // no prompt supporting now
    // This feature is not supported yet, setting it to nullptr temporarily.
    param.prefix_soft_prompt_lengths = nullptr;
    param.p_prompt_tuning_prompt_lengths = nullptr;
    param.max_input_without_prompt_length = maxInputLength;
    param.max_prefix_soft_prompt_length = 0;

    outputIds->reshape(ITensor::makeShape({batchSize, beamWidth, finalSeqLength}));
    param.output_ids = bufferCast<SizeType>(*outputIds);
    param.stream = stream.get();
    tensorrt_llm::kernels::invokeGatherTree(param);
}

// follows https://github.com/NVIDIA/TensorRT/blob/release/8.6/samples/common/sampleEngines.cpp
std::vector<uint8_t> tensorrt_llm::runtime::GptSession::loadEngine(std::string const& enginePath)
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

nvinfer1::ILogger& tensorrt_llm::runtime::GptSession::getLogger() const
{
    return *mLogger;
}

BufferManager& tensorrt_llm::runtime::GptSession::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

int GptSession::initDevice(WorldConfig const& worldConfig)
{
    auto const device = worldConfig.getDevice();
    TLLM_CUDA_CHECK(cudaSetDevice(device));
    return device;
}

void GptSession::RuntimeBuffers::clear()
{
    logits = nullptr;
    sequenceLengths = nullptr;
    pastKeyValueLengths = nullptr;
    tokenMask = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;
    lastTokenIds = nullptr;

    presentKeysVals.clear();
    presentKeysValsAlt.clear();

    keyValuePointers.clear();
    beamWidths = nullptr;
    requestCacheMaxSeqLengths = nullptr;
    hostInputLengths = nullptr;
    cacheIndirectionPointers = nullptr;

    allocated = false;
}
