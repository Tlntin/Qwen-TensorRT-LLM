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

class TrtGptModel
{
public:
    using RequestTable = std::map<uint64_t, LlmRequest>;

    TrtGptModel() {}

    /// @brief Function that advances all requests provided in the table
    /// @param request_table  The set of request for which to update the state
    virtual void forward(RequestTable& request_table) = 0;
};

} // namespace batch_manager
} // namespace inflight_batcher
