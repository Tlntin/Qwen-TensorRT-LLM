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
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <memory>
#include <type_traits>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class TllmBuffersTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP();
    }

    void TearDown() override {}

    int mDeviceCount;
};

TEST_F(TllmBuffersTest, Stream)
{
    CudaStream stream{};
    EXPECT_NE(stream.get(), nullptr);
    auto ptr = std::make_shared<CudaStream>();
    EXPECT_NE(ptr->get(), nullptr);
    EXPECT_GE(ptr->getDevice(), 0);
    CudaStream lease{ptr->get(), ptr->getDevice(), false};
    EXPECT_EQ(lease.get(), ptr->get());
}

TEST_F(TllmBuffersTest, CudaAllocator)
{
    auto constexpr size = 1024;
    CudaAllocator allocator{};
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

TEST_F(TllmBuffersTest, PinnedAllocator)
{
    auto constexpr size = 1024;
    PinnedAllocator allocator{};
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kPINNED);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

TEST_F(TllmBuffersTest, HostAllocator)
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kCPU);
}

TEST_F(TllmBuffersTest, CudaAllocatorAsync)
{
    auto streamPtr = std::make_shared<CudaStream>();
    auto constexpr size = 1024;
    CudaAllocatorAsync allocator{streamPtr};
    auto ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(allocator.deallocate(ptr, size));
    EXPECT_EQ(allocator.getMemoryType(), MemoryType::kGPU);
    streamPtr->synchronize();
    CudaAllocatorAsync allocatorCopy = allocator;
    EXPECT_EQ(allocatorCopy.getCudaStream(), streamPtr);
    CudaAllocatorAsync allocatorMove = std::move(allocatorCopy);
    EXPECT_EQ(allocatorMove.getCudaStream(), streamPtr);
    EXPECT_THROW(allocator.deallocate(ptr, size), std::runtime_error);
}

namespace
{
void testBuffer(IBuffer& buffer, std::int32_t typeSize)
{
    auto const size = buffer.getSize();
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.getSizeInBytes(), size * typeSize);
    EXPECT_EQ(buffer.getCapacity(), size);
    buffer.resize(size / 2);
    EXPECT_EQ(buffer.getSize(), size / 2);
    EXPECT_EQ(buffer.getCapacity(), size);
    buffer.resize(size * 2);
    EXPECT_EQ(buffer.getSize(), size * 2);
    EXPECT_EQ(buffer.getCapacity(), size * 2);
    buffer.release();
    EXPECT_EQ(buffer.getSize(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
    buffer.resize(size / 2);
    EXPECT_EQ(buffer.getCapacity(), size / 2);
    auto bufferWrapped = IBuffer::wrap(buffer.data(), buffer.getDataType(), buffer.getSize(), buffer.getCapacity());
    EXPECT_EQ(bufferWrapped->data(), buffer.data());
    EXPECT_EQ(bufferWrapped->getSize(), buffer.getSize());
    EXPECT_EQ(bufferWrapped->getCapacity(), buffer.getCapacity());
    EXPECT_EQ(bufferWrapped->getDataType(), buffer.getDataType());
    EXPECT_EQ(bufferWrapped->getMemoryType(), buffer.getMemoryType());
    EXPECT_NO_THROW(bufferWrapped->resize(buffer.getCapacity() / 2));
    EXPECT_THROW(bufferWrapped->resize(buffer.getCapacity() * 2), std::bad_alloc);
    auto tensorWrapped = ITensor::wrap(buffer.data(), buffer.getDataType(),
        ITensor::makeShape({static_cast<SizeType>(buffer.getSize())}), buffer.getCapacity());
    EXPECT_EQ(tensorWrapped->getSize(), buffer.getSize());
    EXPECT_EQ(tensorWrapped->getCapacity(), buffer.getCapacity());
    EXPECT_EQ(tensorWrapped->getDataType(), buffer.getDataType());
    EXPECT_EQ(tensorWrapped->getMemoryType(), buffer.getMemoryType());
    EXPECT_NO_THROW(tensorWrapped->reshape(ITensor::makeShape({static_cast<SizeType>(buffer.getCapacity()) / 2})));
    EXPECT_THROW(
        tensorWrapped->reshape(ITensor::makeShape({static_cast<SizeType>(buffer.getCapacity()) * 2})), std::bad_alloc);
}
} // namespace

TEST_F(TllmBuffersTest, DeviceBuffer)
{
    auto streamPtr = std::make_shared<CudaStream>();
    auto constexpr size = 1024;
    CudaAllocatorAsync allocator{streamPtr};
    {
        DeviceBuffer buffer{size, nvinfer1::DataType::kFLOAT, allocator};
        testBuffer(buffer, sizeof(float));
    }
    streamPtr->synchronize();

    static_assert(!std::is_copy_constructible<DeviceBuffer>::value);
    static_assert(!std::is_copy_assignable<DeviceBuffer>::value);
}

TEST_F(TllmBuffersTest, DeviceTensor)
{
    auto streamPtr = std::make_shared<CudaStream>();
    nvinfer1::Dims constexpr dims{3, 16, 8, 4};
    CudaAllocatorAsync allocator{streamPtr};
    {
        DeviceTensor tensor{dims, nvinfer1::DataType::kFLOAT, allocator};
        EXPECT_EQ(tensor.getSize(), ITensor::volume(dims));
        testBuffer(tensor, sizeof(float));
        EXPECT_EQ(tensor.getSize(), ITensor::volume(tensor.getShape()));
    }
    streamPtr->synchronize();

    static_assert(!std::is_copy_constructible<DeviceBuffer>::value);
    static_assert(!std::is_copy_assignable<DeviceBuffer>::value);
}

TEST_F(TllmBuffersTest, BufferSlice)
{
    auto constexpr size = 1024;
    HostAllocator allocator{};
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    auto buffer = std::make_shared<HostBuffer>(size, dataType, allocator);
    auto offset = size / 8;
    auto slice = IBuffer::slice(buffer, offset);
    auto const sizeSlice = size - offset;
    EXPECT_EQ(slice->getSize(), sizeSlice);
    EXPECT_EQ(slice->getCapacity(), sizeSlice);
    EXPECT_EQ(static_cast<std::uint8_t*>(slice->data()) - static_cast<std::uint8_t*>(buffer->data()),
        offset * sizeOfType(dataType));

    EXPECT_NO_THROW(slice->resize(sizeSlice));
    EXPECT_NO_THROW(slice->resize(sizeSlice / 2));
    EXPECT_THROW(slice->resize(sizeSlice * 2), std::invalid_argument);
    EXPECT_NO_THROW(slice->release());
    EXPECT_EQ(slice->data(), nullptr);
}

TEST_F(TllmBuffersTest, TensorSlice)
{
    auto dims = ITensor::makeShape({16, 8, 4});
    HostAllocator allocator{};
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    auto tensor = std::make_shared<HostTensor>(dims, dataType, allocator);
    auto offset = dims.d[0] / 4;
    auto slice = ITensor::slice(tensor, offset);
    auto const sizeSlice = 3 * tensor->getSize() / 4;
    EXPECT_EQ(slice->getShape().d[0], dims.d[0] - offset);
    EXPECT_EQ(slice->getSize(), sizeSlice);
    EXPECT_EQ(slice->getCapacity(), sizeSlice);
    EXPECT_EQ(static_cast<std::uint8_t*>(slice->data()) - static_cast<std::uint8_t*>(tensor->data()),
        offset * ITensor::volume(dims) / dims.d[0] * sizeOfType(dataType));

    auto dimsNew = ITensor::makeShape({12, 32});
    EXPECT_EQ(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[1], dimsNew.d[1]);
    dimsNew.d[0] = 6;
    EXPECT_LT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[0], dimsNew.d[0]);
    dimsNew.d[0] = 16;
    EXPECT_GT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_THROW(slice->reshape(dimsNew), std::invalid_argument);

    EXPECT_NO_THROW(slice->resize(sizeSlice));
    EXPECT_NO_THROW(slice->resize(sizeSlice / 2));
    EXPECT_EQ(slice->getShape().d[0], sizeSlice / 2);
    EXPECT_THROW(slice->resize(sizeSlice * 2), std::invalid_argument);
    EXPECT_NO_THROW(slice->release());
    EXPECT_EQ(slice->data(), nullptr);
}
