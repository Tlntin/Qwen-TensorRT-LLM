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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nvinfer1
{
class IExecutionContext;
}

namespace tensorrt_llm::runtime
{

class ITensor : virtual public IBuffer
{
public:
    using UniquePtr = std::unique_ptr<ITensor>;
    using SharedPtr = std::shared_ptr<ITensor>;
    using UniqueConstPtr = std::unique_ptr<ITensor const>;
    using SharedConstPtr = std::shared_ptr<ITensor const>;

    //!
    //! \brief Returns the tensor dimensions.
    //!
    [[nodiscard]] virtual nvinfer1::Dims const& getShape() const = 0;

    //!
    //! \brief Sets the tensor dimensions. The new size of the tensor will be `volume(dims)`
    //!
    virtual void reshape(nvinfer1::Dims const& dims) = 0;

    ~ITensor() override = default;

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor(ITensor const&) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor& operator=(ITensor const&) = delete;

    //!
    //! \brief Returns the volume of the dimensions. Returns -1 if `d.nbDims < 0`.
    //!
    static std::int64_t volume(nvinfer1::Dims const& dims)
    {
        {
            return dims.nbDims < 0 ? -1
                : dims.nbDims == 0
                ? 0
                : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1}, std::multiplies<>{});
        }
    }

    //!
    //! \brief Returns the volume of the dimensions. Throws `std::invalic_argument` if `d.nbDims < 0`.
    //!
    static std::size_t volumeNonNegative(nvinfer1::Dims const& shape)
    {
        auto const vol = volume(shape);
        TLLM_CHECK_WITH_INFO(0 <= vol, "Invalid tensor shape");
        return static_cast<std::size_t>(vol);
    }

    //!
    //! \brief Creates a sliced view on the underlying `tensor`. The view will have the same data type as `tensor`.
    //!
    //! \param tensor The tensor to view.
    //! \param offset The offset of the view w.r.t. dimension 0 of the tensor.
    //! \param size The size of the view w.r.t. dimension 0 of the tensor.
    //! \return A view on the `buffer`.
    //!
    static UniquePtr slice(SharedPtr const& tensor, std::size_t offset, std::size_t size);

    static UniquePtr slice(SharedPtr const& tensor, std::size_t offset)
    {
        auto const dims = tensor->getShape();
        auto const size = (dims.nbDims > 0 ? dims.d[0] : 0) - offset;
        return slice(tensor, offset, size);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr slice(std::shared_ptr<T> const& tensor, std::size_t offset, std::size_t size)
    {
        return slice(std::const_pointer_cast<ITensor>(tensor), offset, size);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr slice(std::shared_ptr<T> const& tensor, std::size_t offset)
    {
        return slice(std::const_pointer_cast<ITensor>(tensor), offset);
    }

    //!
    //! \brief Returns a view on the underlying `tensor`. This view can be reshaped independently from the underlying
    //! tensor, but not beyond its capacity.
    //!
    static UniquePtr view(SharedPtr const& tensor)
    {
        auto constexpr offset = 0;
        return slice(tensor, offset);
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr view(std::shared_ptr<T> const& tensor)
    {
        return view(std::const_pointer_cast<ITensor>(tensor));
    }

    static UniquePtr view(SharedPtr const& tensor, nvinfer1::Dims dims)
    {
        auto v = view(tensor);
        v->reshape(dims);
        return v;
    }

    template <typename T, std::enable_if_t<std::is_same_v<T, SharedConstPtr::element_type>, int> = 0>
    static UniqueConstPtr view(std::shared_ptr<T> const& tensor, nvinfer1::Dims dims)
    {
        auto v = view(std::const_pointer_cast<ITensor>(tensor));
        v->reshape(dims);
        return v;
    }

    //!
    //! \brief Wraps the given `data` in an `ITensor`. The `ITensor` will not own the underlying `data` and cannot
    //! be reshaped beyond `capacity`.
    //!
    //! \param data The data to wrap.
    //! \param type The data type of the `data`.
    //! \param shape The shape of the tensor.
    //! \param capacity The capacity of the buffer.
    //! \return An `ITensor`.
    static UniquePtr wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape, std::size_t capacity);

    static UniquePtr wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape)
    {
        return wrap(data, type, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(T* data, nvinfer1::Dims const& shape, std::size_t capacity)
    {
        return wrap(data, TRTDataType<T>::value, shape, capacity);
    }

    template <typename T>
    static UniquePtr wrap(T* data, nvinfer1::Dims const& shape)
    {
        return wrap<T>(data, shape, volumeNonNegative(shape));
    }

    template <typename T>
    static UniquePtr wrap(std::vector<T>& v, nvinfer1::Dims const& shape)
    {
        return wrap<T>(v.data(), shape, v.capacity());
    }

    //!
    //! \brief A convenience function to create a tensor shape with the given dimensions.
    //!
    static nvinfer1::Dims makeShape(std::initializer_list<SizeType> const& dims);

    //!
    //! \brief A convenience function for converting a tensor shape to a `string`.
    //!
    static std::string toString(nvinfer1::Dims const& dims);

protected:
    ITensor() = default;
};

} // namespace tensorrt_llm::runtime
