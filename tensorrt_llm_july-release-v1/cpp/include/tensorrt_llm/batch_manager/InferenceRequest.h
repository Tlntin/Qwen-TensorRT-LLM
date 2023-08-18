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

#include "tensorrt_llm/batch_manager/Tensor.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace inflight_batcher
{
namespace common
{

class InferenceRequest
{
public:
    InferenceRequest(std::map<std::string, Tensor>& input_tensors, uint64_t correlation_id)
    {
        std::copy(input_tensors.begin(), input_tensors.end(), std::inserter(input_tensors_, input_tensors_.end()));
        correlation_id_ = correlation_id;
    }

    ~InferenceRequest() {}

    // Deprecate. InferenceRequest should store all input and output
    // tensors.
    std::map<std::string, Tensor>& get_input_tensors()
    {
        return input_tensors_;
    }

    std::map<std::string, Tensor>& get_tensors()
    {
        return input_tensors_;
    }

    uint64_t get_correlation_id()
    {
        return correlation_id_;
    }

private:
    std::map<std::string, Tensor> input_tensors_;
    uint64_t correlation_id_;
};

} // namespace common
} // namespace inflight_batcher
