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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatchInterface.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

namespace decoder_batch
{
class Request
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Request(TensorPtr ids, SizeType _maxNewTokens)
        : ids(std::move(ids))
        , maxNewTokens(_maxNewTokens)
    {
    }

    // mandatory parameters
    TensorPtr ids;         // [inputSeqLen], the input sequence of token ids, on gpu
    SizeType maxNewTokens; // maximum number of tokens to generate for this request

    // optional parameters
    TensorPtr embeddingBias; // [vocabSizePadded], on gpu
    TensorPtr badWordsList;  // [2, badWordsLength], on gpu
    TensorPtr stopWordsList; // [2, stopWordsLength], on gpu
};

class Input
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Input(TensorPtr logits)
        : logits{std::move(logits)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
    }

    // mandatory parameters
    TensorPtr logits; // [batchSize, beamWidth, vocabSizePadded], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, beamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, beamWidth, maxSeqLen], mandatory in beam search, on gpu
};
} // namespace decoder_batch

class IGptDecoderBatch;

//! GPT decoder class with support for in-flight batching
class GptDecoderBatch : public IGptDecoderBatch
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoderBatch(
        std::size_t vocabSize, std::size_t vocabSizePadded, SizeType endId, SizeType padId, CudaStreamPtr stream);

    //! Setup the decoder before calling `forward()`
    void setup(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

    //! @brief Initialize the decoder at `batchIdx` with a new `request`.
    void newRequest(SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig);

    //! @brief Run one step for all requests.
    //! Note that this method will synchronize with the stream associated with the decoder.
    void forward(decoder_batch::Output& output, decoder_batch::Input const& input);

    //! @return [batchSize], indicators of finished requests
    [[nodiscard]] std::vector<bool> const& getFinished() const
    {
        return mFinished;
    }

    //! @brief Mark request `batchIdx` as finished.
    void setFinished(SizeType batchIdx)
    {
        mFinished[batchIdx] = true;
    }

    //! @returns [batchSize, beamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token ids
    //! without padding, on gpu
    [[nodiscard]] TensorPtr const& getOutputIds() const
    {
        return mOutputIds;
    }

    //! @returns [batchSize, beamWidth, maxInputLength + maxNewTokens], contains parent ids collected during beam search
    //! without padding, on gpu
    [[nodiscard]] TensorPtr const& getParentIds() const
    {
        return mOutputParentIds;
    }

    //! @returns [batchSize, beamWidth], marks finished requests (per beam), on gpu
    [[nodiscard]] TensorPtr const& getFinishedBeams() const
    {
        return mFinishedBeams;
    }

    //! @returns [batchSize, beamWidth], total sequence lengths (per beam), on gpu
    [[nodiscard]] TensorPtr const& getOutputLengths() const
    {
        return mOutputLengths;
    }

    //! @returns [batchSize, beamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr const& getNewTokens() const
    {
        return mNewTokens;
    }

    //! @returns [batchSize], the number of generation steps executed on each request
    [[nodiscard]] std::vector<SizeType> const& getNbSteps() const
    {
        return mNbSteps;
    }

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    SizeType const mEndId;
    SizeType const mPadId;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;
    tensorrt_llm::common::EventPtr mEventStart, mEventStop;

    std::vector<CudaStreamPtr> mStreams;
    std::vector<tensorrt_llm::common::EventPtr> mEvents;
    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    std::vector<GptDecoderPtr> mDecoders;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    std::vector<DecodingInputPtr> mDecodingInputs;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    std::vector<DecodingOutputPtr> mDecodingOutputs;
    std::vector<SizeType> mNbSteps;
    std::vector<bool> mFinished;
    std::vector<SizeType> mMaxNewTokens;
    SizeType mMaxSequenceLength{};

    // Shared decoding outputs
    TensorPtr mOutputIds;
    TensorPtr mOutputParentIds;
    TensorPtr mFinishedBeams;
    TensorPtr mOutputLengths;
    TensorPtr mCumLogProbs;

    // Shared decoding inputs
    TensorPtr mDummyLogits;
    TensorPtr mEndIds;
    TensorPtr mSequenceLimitLength;
    TensorPtr mInputLengths;

    // Convenience buffer
    TensorPtr mNewTokens;
};
} // namespace tensorrt_llm::runtime
