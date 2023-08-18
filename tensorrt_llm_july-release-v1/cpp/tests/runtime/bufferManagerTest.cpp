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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <limits>
#include <memory>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class BufferManagerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount > 0)
        {
            mStream = std::make_unique<CudaStream>();
        }
        else
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}

    int mDeviceCount;
    BufferManager::CudaStreamPtr mStream;
};

namespace
{

template <typename T>
T convertType(std::size_t val)
{
    return static_cast<T>(val);
}

template <>
half convertType(std::size_t val)
{
    return __float2half_rn(static_cast<float>(val));
}

template <typename T>
void testRoundTrip(BufferManager& manager)
{
    auto constexpr size = 128;
    std::vector<T> inputCpu(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        inputCpu[i] = convertType<T>(i);
    }
    auto inputGpu = manager.copyFrom(inputCpu, MemoryType::kGPU);
    auto outputCpu = manager.copyFrom(*inputGpu, MemoryType::kPINNED);
    EXPECT_EQ(inputCpu.size(), outputCpu->getSize());
    manager.getStream().synchronize();
    auto outputCpuTyped = bufferCast<T>(*outputCpu);
    for (size_t i = 0; i < inputCpu.size(); ++i)
    {
        EXPECT_EQ(inputCpu[i], outputCpuTyped[i]);
    }

    manager.setZero(*inputGpu);
    manager.copy(*inputGpu, *outputCpu);
    manager.getStream().synchronize();
    for (size_t i = 0; i < inputCpu.size(); ++i)
    {
        EXPECT_EQ(0, static_cast<int32_t>(outputCpuTyped[i]));
    }
}
} // namespace

TEST_F(BufferManagerTest, CreateCopyRoundTrip)
{
    BufferManager manager(mStream);
    testRoundTrip<float>(manager);
    testRoundTrip<half>(manager);
    testRoundTrip<std::int8_t>(manager);
    testRoundTrip<std::uint8_t>(manager);
    testRoundTrip<std::int32_t>(manager);
}

TEST_F(BufferManagerTest, MemPoolAttributes)
{
    BufferManager manager(mStream); // sets attributes of the default memory pool
    auto const device = mStream->getDevice();
    ::cudaMemPool_t memPool;
    TLLM_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::uint64_t threshold{0};
    TLLM_CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
    EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());
}
