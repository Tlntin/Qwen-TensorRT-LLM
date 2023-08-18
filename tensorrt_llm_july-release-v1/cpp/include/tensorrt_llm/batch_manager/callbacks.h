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

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "tensorrt_llm/batch_manager/InferenceRequest.h"
#include "tensorrt_llm/batch_manager/Tensor.h"

namespace inflight_batcher
{
namespace common
{

using GetInferenceRequestsCallback = std::function<std::list<std::shared_ptr<InferenceRequest>>(int32_t)>;
using SendResponseCallback = std::function<bool(uint64_t, std::list<std::shared_ptr<Tensor>>, bool)>;

} // namespace common
} // namespace inflight_batcher
