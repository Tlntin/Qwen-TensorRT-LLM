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

#include "tensorrt_llm/batch_manager/batchSlotManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/gptDecoderBatchInterface.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferPlugin.h>

using namespace tensorrt_llm::runtime;

namespace inflight_batcher
{
namespace batch_manager
{

using TensorMap = StringPtrMap<ITensor>;
using TensorPtr = ITensor::SharedPtr;
using ReqIdsVec = std::vector<uint64_t>;

class TrtGptModelInflightBatching : public TrtGptModel
{
public:
    using RequestTable = std::map<uint64_t, LlmRequest>;

    TrtGptModelInflightBatching(int32_t maxSeqLen, int32_t maxNumRequests,
        // TODO: How should beamWidth be handled?
        int32_t beamWidth, std::shared_ptr<nvinfer1::ILogger> logger, GptModelConfig modelConfig,
        WorldConfig worldConfig, std::shared_ptr<TllmRuntime> runtime, std::shared_ptr<IGptDecoderBatch> decoder);

    /// @brief Function that advances all requests provided in the table
    /// @param request_table  The set of request for which to update the state
    void forward(RequestTable& request_table);

    /// @brief Function that advances all requests provided in the table
    void forwardToCompletion(RequestTable& request_table);

    class RuntimeBuffers
    {
    public:
        // general
        TensorPtr inputsIds;
        TensorPtr inputLengths;
        TensorPtr inputOffsets;

        // runtime
        TensorPtr logits;
        TensorPtr batchSlotLogits;
        TensorPtr sequenceLengths;     // with attention plugin
        TensorPtr pastKeyValueLengths; // with attention plugin
        TensorPtr tokenMask;           // with attention plugin
        TensorPtr attentionMask;       // without attention plugin
        TensorPtr positionIds;
        TensorPtr lastTokenIds;
        TensorPtr maxInputLength;
        TensorPtr cacheIndirection0;
        TensorPtr cacheIndirection1;

        // Inputs specific to Inflight Batching Attention plugin
        TensorPtr beamWidths;
        std::vector<TensorPtr> keyValuePointers;
        TensorPtr requestCacheMaxSeqLengths;
        TensorPtr hostInputLengths;
        TensorPtr cacheIndirectionPointers;
        // TensorPtr kvCacheBlockPointers;
        // TensorPtr pointersToKVCacheBlockPointers;

        std::vector<TensorPtr> presentKeysVals;

        // decoder
        // TODO: Should this be used?
        TensorPtr cumLogProbs;
        // TODO: Is this used?
        TensorPtr parentIds;

        bool allocated{false};

        void clear();
    };

private:
    void createContexts();
    void createBuffers();

    void reshapeBuffers(int32_t batchSize, int32_t beamWidth, int32_t maxInputLen);

    void assignReqBatchSlots(const RequestTable& requestTable);

    void executeContext(const RequestTable& requestTable, const ReqIdsVec& reqIds, bool isContextPhase);

    int32_t getMaxInputLen(const RequestTable& requestTable, const ReqIdsVec& reqIds, bool isContextPhase);

    void prepareStep(TensorMap& inputBuffers, TensorMap& outputBuffers, const RequestTable& requestTable,
        const ReqIdsVec& reqIds, bool isContextPhase);

    void getBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, bool isContextPhase);

    void setupDecoderStep(RequestTable& requestTable, const ReqIdsVec& contextReqIds);

    void decoderStep(RequestTable& requestTable);

    template <typename T>
    void setRawPointers(TensorPtr& pointers, TensorPtr const& input, int32_t bid, int32_t batchSlot);

    void zeroBatchSlot(TensorPtr& tensor, int32_t batchSlot);

    void setRuntimeBuffersFromInputs(const RequestTable& requestTable, const ReqIdsVec& reqIds, bool isContextPhase);

    int32_t mMaxSeqLen;
    int32_t mMaxNumRequests;
    int32_t mBeamWidth;
    int32_t mPadId;
    int32_t mEndId;

    std::shared_ptr<nvinfer1::ILogger> mLogger;
    GptModelConfig mModelConfig;
    WorldConfig mWorldConfig;
    std::shared_ptr<TllmRuntime> mRuntime;
    std::shared_ptr<IGptDecoderBatch> mDecoder;

    nvinfer1::IExecutionContext* mContext0;
    nvinfer1::IExecutionContext* mContext1;

    RuntimeBuffers mBuffers;
    TensorPtr mInputOffsets;

    TensorMap mInputBuffers;  // has to persist to keep tensors alive
    TensorMap mOutputBuffers; // has to persist to keep tensors alive

    int32_t mContextIdFlipFlop;

    TensorPtr mDecoderOutputCacheIndirection;
    TensorPtr mDecoderInputCacheIndirection;

    std::map<int64_t, int32_t> mReqIdToBatchSlot;
    std::shared_ptr<BatchSlotManager> mBatchSlotManager;
};

} // namespace batch_manager
} // namespace inflight_batcher
