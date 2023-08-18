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

#include <NvInferRuntime.h>

#include <cstdint>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace tensorrt_llm::runtime
{

enum class MemoryType : std::int32_t
{
    kGPU = 0,
    kCPU = 1,
    kPINNED = 2
};

template <typename T>
struct TRTDataType
{
};

template <>
struct TRTDataType<float>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kFLOAT;
};

template <>
struct TRTDataType<half>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kHALF;
};

template <>
struct TRTDataType<std::int8_t>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kINT8;
};

template <>
struct TRTDataType<std::int32_t>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kINT32;
};

template <>
struct TRTDataType<std::int64_t>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kINT64;
};

template <>
struct TRTDataType<bool>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kBOOL;
};

template <>
struct TRTDataType<std::uint8_t>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TRTDataType<__nv_bfloat16>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TRTDataType<__nv_fp8_e4m3>
{
    static constexpr nvinfer1::DataType value = nvinfer1::DataType::kFP8;
};
#endif

class IBuffer
{
public:
    using UniquePtr = std::unique_ptr<IBuffer>;
    using SharedPtr = std::shared_ptr<IBuffer>;
    using UniqueConstPtr = std::unique_ptr<IBuffer const>;
    using SharedConstPtr = std::shared_ptr<IBuffer const>;

    //!
    //! \brief Returns pointer to underlying array.
    //!
    virtual void* data() = 0;

    //!
    //! \brief Returns pointer to underlying array.
    //!
    virtual void const* data() const = 0;

    //!
    //! \brief Returns the size (in number of elements) of the buffer.
    //!
    virtual std::size_t getSize() const = 0;

    //!
    //! \brief Returns the size (in bytes) of the buffer.
    //!
    virtual std::size_t getSizeInBytes() const = 0;

    //!
    //! \brief Returns the capacity of the buffer.
    //!
    virtual std::size_t getCapacity() const = 0;

    //!
    //! \brief Returns the data type of the buffer.
    //!
    virtual nvinfer1::DataType getDataType() const = 0;

    //!
    //! \brief Returns the memory type of the buffer.
    //!
    virtual MemoryType getMemoryType() const = 0;

    //!
    //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
    //!
    virtual void resize(std::size_t newSize) = 0;

    //!
    //! \brief Releases the buffer. It will be reset to nullptr.
    //!
    virtual void release() = 0;

    virtual ~IBuffer() = default;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer(IBuffer const&) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer& operator=(IBuffer const&) = delete;

    //!
    //! \brief Creates a sliced view on the underlying `buffer`. The view will have the same data type as `buffer`.
    //!
    //! \param buffer The buffer to view.
    //! \param offset The offset of the view.
    //! \param size The size of the view.
    //! \return A view on the `buffer`.
    //!
    static UniquePtr slice(SharedPtr const& buffer, std::size_t offset, std::size_t size);

    static UniquePtr slice(SharedPtr const& buffer, std::size_t offset)
    {
        auto const size = buffer->getSize() - offset;
        return slice(buffer, offset, size);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr slice(std::shared_ptr<T> const& tensor, std::size_t offset, std::size_t size)
    {
        return slice(std::const_pointer_cast<IBuffer>(tensor), offset, size);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr slice(std::shared_ptr<T> const& tensor, std::size_t offset)
    {
        return slice(std::const_pointer_cast<IBuffer>(tensor), offset);
    }

    //!
    //! \brief Returns a view on the underlying `buffer`. This view can be resized independently from the underlying
    //! buffer, but not beyond its capacity.
    //!
    static UniquePtr view(SharedPtr const& buffer)
    {
        auto constexpr offset = 0;
        return slice(buffer, offset);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr view(std::shared_ptr<T> const& buffer)
    {
        return view(std::const_pointer_cast<IBuffer>(buffer));
    }

    static UniquePtr view(SharedPtr const& buffer, std::size_t size)
    {
        auto v = view(buffer);
        v->resize(size);
        return v;
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr view(std::shared_ptr<T> const& buffer, std::size_t size)
    {
        auto v = view(std::const_pointer_cast<IBuffer>(buffer));
        v->resize(size);
        return v;
    }

    //!
    //! \brief Wraps the given `data` in an `IBuffer`. The `IBuffer` will not own the underlying `data` and cannot
    //! be resized beyond `capacity`.
    //!
    //! \param data The data to wrap.
    //! \param type The data type of the `data`.
    //! \param size The size of the buffer.
    //! \param capacity The capacity of the buffer.
    //! \return An `IBuffer`.
    static UniquePtr wrap(void* data, nvinfer1::DataType type, std::size_t size, std::size_t capacity);

    static UniquePtr wrap(void* data, nvinfer1::DataType type, std::size_t size)
    {
        return wrap(data, type, size, size);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, size, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, std::size_t size)
    {
        return wrap<T>(data, size);
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v)
    {
        return wrap<T>(v.data(), v.size(), v.capacity());
    }

    //!
    //! \brief Determine the memory type of a pointer.
    //!
    static MemoryType memoryType(void const* data);

protected:
    IBuffer() = default;
};

template <typename T>
T const* bufferCast(IBuffer const& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return reinterpret_cast<T const*>(buffer.data());
}

template <typename T>
T* bufferCast(IBuffer& buffer)
{
    if (TRTDataType<typename std::remove_cv<T>::type>::value != buffer.getDataType())
    {
        throw std::bad_cast();
    }
    return reinterpret_cast<T*>(buffer.data());
}

} // namespace tensorrt_llm::runtime
