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

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

namespace inflight_batcher
{
namespace common
{

enum TensorMemory_t
{
    MT_INVALID = 0,
    MT_GPU = 1,
    MT_HOST = 2
};

enum Tensor_t
{
    DT_INVALID = 0,
    DT_BOOL,
    DT_UINT8,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
    DT_INT8,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_FP16,
    DT_FP32,
    DT_FP64,
    DT_BYTES,
    DT_BF16
};

using TensorDataType = std::variant<std::vector<int32_t>, std::vector<float>>;

class Tensor
{
public:
    Tensor(){};
    Tensor(std::string const& name, TensorMemory_t mk, Tensor_t dt, std::vector<int64_t> shape);
    Tensor(const Tensor& obj);
    ~Tensor();

    std::vector<int64_t> shape();
    std::vector<int64_t> stride();

    template <typename T>
    T* data_ptr();

    int64_t size();

    void* void_data_ptr();
    void raw_copy_from(const void* src, int64_t len, int64_t off);
    void raw_copy_to(void* dst, int64_t len, int64_t off);

    const std::string& name() const;
    Tensor_t datatype();
    int64_t sizeBytes();

private:
    std::string name_;
    TensorMemory_t mk_{MT_INVALID};
    Tensor_t dt_{DT_INVALID};
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    int64_t dims_count_;
    int64_t mSize;
    TensorDataType data_;
};

} // namespace common
} // namespace inflight_batcher
