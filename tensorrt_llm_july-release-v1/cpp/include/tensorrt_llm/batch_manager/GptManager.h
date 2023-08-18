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

#include "stdlib.h"
#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/InferenceRequest.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/batch_manager/trtGptModelFactory.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <unordered_map>

using namespace inflight_batcher::common;
using namespace inflight_batcher::batch_manager;
using namespace tensorrt_llm::runtime;

namespace inflight_batcher
{
namespace batch_manager
{

using RequestTable = std::map<uint64_t, LlmRequest>;
using TensorMap = StringPtrMap<ITensor>;
using TensorPtr = ITensor::SharedPtr;

/* Responsible for shepherding Triton requests through to completion
   using TRT Backend. Each Triton backend should have just one of these. */
class GptManager
{
public:
    GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, int32_t mMaxSeqLen,
        int32_t maxNumRequests, GetInferenceRequestsCallback getInferenceRequestsCb,
        SendResponseCallback sendResponseCb);

    /* Wraps the user-provided callback for requests.
       Adds requests to request table.
       Invoked every generation loop iteration. */
    BatchManagerErrorCode_t fetch_new_requests();

    /* Does the following:
       1. Returns completed requests to Triton
       2. Frees KV cache and other dedicated resources
       3. Deletes entry from request_table */
    BatchManagerErrorCode_t return_completed_requests();

    virtual ~GptManager();

protected:
    /* Does the following:
       1. Maps batch manager requests to backend request
       2. Invokes one step of backend
       3. Updates state of all requests */
    virtual BatchManagerErrorCode_t step(RequestTable& requestTable);

private:
    static LlmRequest fillLlmRequest(std::shared_ptr<InferenceRequest> newReq);

    static BatchManagerErrorCode_t getReqInputTokens(
        std::shared_ptr<InferenceRequest> new_req, std::vector<int32_t>& tokens);
    static BatchManagerErrorCode_t getMaxNewTokens(std::shared_ptr<InferenceRequest> newReq, int32_t& maxNewTokens);

    std::shared_ptr<TrtGptModel> mTrtGptModel;
    int32_t mMaxSeqLen;
    int32_t mMaxNumRequests;
    // Table of live requests
    std::map<uint64_t, LlmRequest> mRequestTable;

    GetInferenceRequestsCallback mGetInferenceRequestsCb;
    SendResponseCallback mSendResponseCb;

    std::atomic<bool> destructor_called_;
    void decoupled_execution_loop();
    std::shared_ptr<std::thread> worker_thread_;
    inline static const std::string kInputIdsTensorName_ = "input_ids";
    inline static const std::string kMaxNewTokensTensorName_ = "request_output_len";
    inline static const std::string kOutputIdsTensorName_ = "output_ids";

    std::shared_ptr<nvinfer1::ILogger> mLogger{};
    std::shared_ptr<GptSession> mSession;

    // Get an unused KV cache
    // int32_t get_kv_cache();

    // Release existing KV cache
    // void free_kv_cache();
};

} // namespace batch_manager
} // namespace inflight_batcher
