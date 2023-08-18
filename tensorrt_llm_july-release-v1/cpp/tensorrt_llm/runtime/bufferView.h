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

#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <memory>
#include <string>

namespace tensorrt_llm::runtime
{
class BufferView : virtual public IBuffer
{
public:
    explicit BufferView(IBuffer::SharedPtr buffer, std::size_t offset, std::size_t size)
        : mBuffer(std::move(buffer))
        , mOffset{offset}
        , mSize{size}
    {
        if (offset > mBuffer->getSize())
        {
            throw std::out_of_range(std::string("offset ") + std::to_string(offset) + " exceeds buffer size "
                + std::to_string(mBuffer->getSize()));
        }

        if (offset + size > mBuffer->getSize())
        {
            throw std::out_of_range(std::string("slice ") + std::to_string(offset + size) + " exceeds buffer size "
                + std::to_string(mBuffer->getSize()));
        }
    }

    void* data() override
    {
        if (getSize() == 0)
            return nullptr;
        auto* data = mBuffer->data();
        return data ? static_cast<std::uint8_t*>(data) + sizeInBytes(mOffset) : nullptr;
    }

    [[nodiscard]] void const* data() const override
    {
        if (getSize() == 0)
            return nullptr;
        auto* data = mBuffer->data();
        return data ? static_cast<std::uint8_t const*>(data) + sizeInBytes(mOffset) : nullptr;
    }

    [[nodiscard]] size_t getSize() const override
    {
        return mSize;
    }

    [[nodiscard]] size_t getSizeInBytes() const override
    {
        return sizeInBytes(mSize);
    }

    [[nodiscard]] size_t getCapacity() const override
    {
        return mBuffer->getCapacity() - mOffset;
    }

    [[nodiscard]] nvinfer1::DataType getDataType() const override
    {
        return mBuffer->getDataType();
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mBuffer->getMemoryType();
    }

    void resize(std::size_t newSize) override
    {
        if (newSize > getCapacity())
        {
            throw std::invalid_argument("BufferView::resize() exceeding capacity is not supported");
        }
        mSize = newSize;
    }

    void release() override
    {
        mSize = 0;
    }

    ~BufferView() override = default;

private:
    IBuffer::SharedPtr mBuffer;
    std::size_t mOffset, mSize;

    [[nodiscard]] std::size_t sizeInBytes(std::size_t size) const
    {
        return size * sizeOfType(getDataType());
    }
};

} // namespace tensorrt_llm::runtime
