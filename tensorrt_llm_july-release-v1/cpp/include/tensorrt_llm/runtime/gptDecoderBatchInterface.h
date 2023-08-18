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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

namespace decoder_batch
{
class Request;
class Output;
class Input;
} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class IGptDecoderBatch
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using TensorConstPtr = std::shared_ptr<ITensor const>;

    IGptDecoderBatch(){};
    IGptDecoderBatch(
        std::size_t vocabSize, std::size_t vocabSizePadded, SizeType endId, SizeType padId, CudaStreamPtr stream){};

    //! Setup the decoder before calling `forward()`
    virtual void setup(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength) = 0;

    //! @brief Initialize the decoder at `batchIdx` with a new `request`.
    virtual void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
        = 0;

    //! @brief Run one step for all requests.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    virtual std::vector<bool> const& getFinished() const = 0;

    virtual void setFinished(SizeType batchIdx) = 0;

    virtual TensorPtr const& getOutputIds() const = 0;

    //! @returns [batchSize, beamWidth], marks finished requests (per beam), on gpu
    virtual TensorPtr const& getFinishedBeams() const = 0;

    //! @returns [batchSize, beamWidth], total sequence lengths (per beam), on gpu
    virtual TensorPtr const& getOutputLengths() const = 0;

    virtual TensorPtr const& getNewTokens() const = 0;

    virtual std::vector<SizeType> const& getNbSteps() const = 0;
};

} // namespace tensorrt_llm::runtime
