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

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include <assert.h>
#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

using namespace inflight_batcher::common;
using namespace tensorrt_llm::runtime;

namespace inflight_batcher
{
namespace batch_manager
{

enum LlmRequestState_t
{
    REQUEST_STATE_UNKNOWN = 0,
    REQUEST_STATE_CONTEXT_INIT = 1,
    REQUEST_STATE_GENERATION_IN_PROGRESS = 2,
    REQUEST_STATE_GENERATION_COMPLETE = 3
};

class LlmRequest
{
public:
    LlmRequest(uint64_t requestId, int32_t maxNewTokens, std::shared_ptr<std::vector<int32_t>> tokens,
        SamplingConfig samplingConfig)
        : mRequestId(requestId)
        , mMaxNewTokens(maxNewTokens)
        , mTokens(tokens)
        , mSamplingConfig(samplingConfig)
        , mState(REQUEST_STATE_CONTEXT_INIT)
        , mPromptLen(tokens->size())
        , mNumGeneratedTokens(0)
    {
    }

    uint64_t mRequestId;
    int32_t mMaxNewTokens;
    std::shared_ptr<std::vector<int32_t>> mTokens;
    SamplingConfig mSamplingConfig;
    int32_t mPromptLen;
    int32_t mNumGeneratedTokens;
    LlmRequestState_t mState;

    ~LlmRequest() {}
};

} // namespace batch_manager
} // namespace inflight_batcher
