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
#include "tensorrt_llm/runtime/gptDecoderBatch.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

GptDecoderBatch::GptDecoderBatch(std::size_t vocabSize, std::size_t vocabSizePadded, SizeType endId, SizeType padId,
    GptDecoderBatch::CudaStreamPtr stream)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mEndId{endId}
    , mPadId{padId}
    , mStream{std::move(stream)}
    , mBufferManager{mStream}
    , mEventStart(tc::CreateEvent())
    , mEventStop(tc::CreateEvent())
{
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    mOutputIds = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType));
    mOutputParentIds = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType));
    mFinishedBeams = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value));
    mOutputLengths = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType));
    mCumLogProbs = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType));

    mEndIds = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType));
    mSequenceLimitLength = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType));
    mInputLengths = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType));

    mDummyLogits = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType));

    mNewTokens = std::shared_ptr(mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType));
}

void GptDecoderBatch::setup(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength)
{
    TLLM_CHECK(batchSize > 0);
    TLLM_CHECK(beamWidth > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mMaxSequenceLength = maxSequenceLength;
    auto const batchSizeXbeamWidth = ITensor::makeShape({batchSize, beamWidth});
    mOutputIds->reshape(ITensor::makeShape({batchSize, beamWidth, maxSequenceLength}));
    mOutputParentIds->reshape(ITensor::makeShape({batchSize, beamWidth, maxSequenceLength}));
    mFinishedBeams->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*mFinishedBeams);
    mOutputLengths->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*mOutputLengths);
    mCumLogProbs->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*mCumLogProbs);

    std::vector<std::int32_t> endIdsVec(batchSize * beamWidth, mEndId);
    mEndIds->reshape(batchSizeXbeamWidth);
    kernels::invokeFill(*mEndIds, mEndId, *mStream);
    mSequenceLimitLength->reshape(ITensor::makeShape({batchSize}));
    kernels::invokeFill(*mSequenceLimitLength, mMaxSequenceLength, *mStream);
    mInputLengths->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*mInputLengths);

    mNewTokens->reshape(batchSizeXbeamWidth);

    mStreams.resize(batchSize);
    mEvents.resize(batchSize);
    mDecoders.resize(batchSize);
    mDecodingInputs.resize(batchSize);
    mDecodingOutputs.resize(batchSize);
    mNbSteps.resize(batchSize);
    mFinished.resize(batchSize);
    mMaxNewTokens.resize(batchSize);
    auto const device = mStream->getDevice();
    for (SizeType i = 0; i < batchSize; ++i)
    {
        mStreams[i] = std::make_shared<CudaStream>(device);
        mEvents[i] = tc::CreateEvent();
        mDecoders[i] = IGptDecoder::create(TRTDataType<float>::value, mVocabSize, mVocabSizePadded, mStreams[i]);
        mDecodingInputs[i].reset();
        mDecodingOutputs[i].reset();
        mNbSteps[i] = 0;
        mFinished[i] = true;
        mMaxNewTokens[i] = 0;
    }
}

void GptDecoderBatch::newRequest(
    SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_CHECK(batchIdx >= 0);
    auto const& jointOutputIdsShape = mOutputIds->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(0 <= batchSize && batchIdx < batchSize);
    auto const beamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK_WITH_INFO(samplingConfig.beamWidth == beamWidth, "Beam width must be equal for all requests.");
    auto const maxInputLength = mMaxSequenceLength - request.maxNewTokens;
    auto const inputLength = request.ids->getShape().d[0];
    TLLM_CHECK_WITH_INFO(inputLength <= maxInputLength, "Input length must be less than max input length.");
    TLLM_CHECK(request.ids->getDataType() == TRTDataType<TokenIdType>::value);

    auto constexpr localBatchSize = 1;

    auto& stream = mStreams[batchIdx];
    BufferManager manager{stream};

    // input
    auto& input = mDecodingInputs.at(batchIdx);
    input = std::make_unique<DecodingInput>(
        inputLength, inputLength, localBatchSize, mDummyLogits, ITensor::slice(mEndIds, batchIdx, localBatchSize));
    input->embeddingBias = request.embeddingBias;
    input->badWordsList = request.badWordsList;
    input->stopWordsList = request.stopWordsList;
    TensorPtr sequenceLimitLength{ITensor::slice(mSequenceLimitLength, batchIdx, localBatchSize)};
    kernels::invokeFill(*sequenceLimitLength, inputLength + request.maxNewTokens, *stream);
    input->sequenceLimitLength = sequenceLimitLength;
    TensorPtr inputLengths{ITensor::slice(mInputLengths, batchIdx, localBatchSize)};
    kernels::invokeFill(*inputLengths, inputLength, *stream);
    input->lengths = inputLengths;

    // output
    auto& output = mDecodingOutputs.at(batchIdx);
    auto const outputIdsShape = ITensor::makeShape({mMaxSequenceLength, localBatchSize, beamWidth});
    auto outputIds = std::shared_ptr(manager.gpu(outputIdsShape, TRTDataType<TokenIdType>::value));
    output = std::make_unique<DecodingOutput>(outputIds);
    output->finished = ITensor::slice(mFinishedBeams, batchIdx, localBatchSize);
    manager.setZero(*output->finished);
    output->finishedSum = BufferManager::pinned(ITensor::makeShape({1}), TRTDataType<SizeType>::value);
    *bufferCast<std::int32_t>(*output->finishedSum) = 0;
    output->lengths = ITensor::slice(mOutputLengths, batchIdx, localBatchSize);
    kernels::invokeFill(*output->lengths, inputLength, *stream);
    output->cumLogProbs = ITensor::slice(mCumLogProbs, batchIdx, localBatchSize);
    manager.setZero(*output->cumLogProbs);

    if (beamWidth > 1)
    {
        // Set all but the first beam to a small value
        kernels::invokeFill(*IBuffer::slice(output->cumLogProbs, 1), -1e20f, *stream);
        output->parentIds = manager.gpu(outputIdsShape, TRTDataType<SizeType>::value);
        manager.setZero(*output->parentIds);
    }

    // remaining
    mDecoders[batchIdx]->setup(samplingConfig, localBatchSize);
    mNbSteps[batchIdx] = 0;
    mFinished[batchIdx] = false;
    mMaxNewTokens[batchIdx] = request.maxNewTokens;

    // copy the request ids into outputIds (with tiling)
    auto lengthsView = ITensor::view(inputLengths, ITensor::makeShape({1}));
    kernels::invokeCopyInputToOutput(*output->ids, *request.ids, *lengthsView, mPadId, *stream);

    // copy the request ids into mOutputIds
    auto inputIdsView = ITensor::view(
        std::const_pointer_cast<ITensor>(request.ids), ITensor::makeShape({localBatchSize, inputLength}));
    auto outputIdsView = ITensor::slice(mOutputIds, batchIdx, localBatchSize);
    outputIdsView->reshape(ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, mPadId, *stream);
    kernels::invokeTileTensor<SizeType>(*outputIdsView, *inputIdsView, beamWidth, *stream);
}

void GptDecoderBatch::forward(decoder_batch::Output& output, decoder_batch::Input const& input)
{
    auto& logits = input.logits;
    TLLM_CHECK(logits->getDataType() == TRTDataType<float>::value);

    auto const& logitsShape = logits->getShape();

    auto const& jointOutputIdsShape = mOutputIds->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(logitsShape.d[0] == batchSize);
    auto const beamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK(logitsShape.d[1] == beamWidth);
    TLLM_CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    TLLM_CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    mStream->record(mEventStart.get());
    for (std::size_t i = 0; i < mDecoders.size(); ++i)
    {
        if (mFinished[i])
            continue;

        auto& stream = mStreams[i];
        stream->wait(mEventStop.get());
        auto& dInput = *mDecodingInputs[i];
        auto& dOutput = *mDecodingOutputs[i];
        dInput.logits = ITensor::slice(logits, i, 1);
        if (srcCacheIndirection && tgtCacheIndirection)
        {
            dInput.cacheIndirection = ITensor::slice(srcCacheIndirection, i, 1);
            dOutput.cacheIndirection = ITensor::slice(tgtCacheIndirection, i, 1);
        }

        auto& decoder = *mDecoders[i];
        decoder.forwardAsync(dOutput, dInput);

        auto outputIdsView = ITensor::slice(dOutput.ids, dInput.step, 1);
        auto const& outputShape = outputIdsView->getShape();
        outputIdsView->reshape(ITensor::makeShape({outputShape.d[1], outputShape.d[2]}));

        auto jointOutputIdsView = ITensor::slice(mOutputIds, i, 1);
        auto const& jointOutputShape = jointOutputIdsView->getShape();
        jointOutputIdsView->reshape(ITensor::makeShape({jointOutputShape.d[1], jointOutputShape.d[2]}));

        kernels::invokeTransposeWithOutputOffset(*jointOutputIdsView, *outputIdsView, dInput.step, *stream);

        if (beamWidth > 1)
        {
            auto outputParentIdsView = ITensor::slice(dOutput.parentIds, dInput.step, 1);
            auto const& outputParentIdsShape = outputParentIdsView->getShape();
            outputParentIdsView->reshape(ITensor::makeShape({outputParentIdsShape.d[1], outputParentIdsShape.d[2]}));

            auto jointOutputParentIdsView = ITensor::slice(mOutputParentIds, i, 1);
            auto const& jointOutputParentIdsShape = jointOutputParentIdsView->getShape();
            jointOutputParentIdsView->reshape(
                ITensor::makeShape({jointOutputParentIdsShape.d[1], jointOutputParentIdsShape.d[2]}));

            kernels::invokeTransposeWithOutputOffset(
                *jointOutputParentIdsView, *outputParentIdsView, dInput.step, *stream);
        }

        auto newTokensView = ITensor::slice(mNewTokens, i, 1);
        BufferManager manager{stream};
        manager.copy(*outputIdsView, *newTokensView);

        auto& event = mEvents[i];
        stream->record(event.get());
        mStream->wait(event.get());
        dInput.step += 1;
        mNbSteps[i] += 1;
    }
    mStream->record(mEventStop.get());
    TLLM_CUDA_CHECK(::cudaEventSynchronize(mEventStop.get()));

    for (std::size_t i = 0; i < mDecoders.size(); ++i)
    {
        auto& dOutput = *mDecodingOutputs[i];
        mFinished[i] = mNbSteps[i] >= mMaxNewTokens[i]
            // This condition requires the synchronization above
            || *bufferCast<SizeType>(*dOutput.finishedSum) == dOutput.finished->getSize();
    }
}
