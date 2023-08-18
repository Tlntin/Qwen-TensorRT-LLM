/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "stdlib.h"
#include "tensor.h"

namespace tensorrt_llm
{
namespace common
{

/* TODO(mseznec): use enum instead of int for quant_mode */
/* enum class QuantMode { */
/*     NoQuant, */
/*     A16W8Quant, */
/*     A8W8Quant, */
/* }; */

class QuantOption
{
public:
    enum Value : int
    {
        PerTensorQuant,
        PerTokenQuant,
        PerChannelQuant,
        PerTokenChannelQuant,
        None
    };

    QuantOption() = default;

    constexpr QuantOption(Value mode)
        : mValue(mode)
    {
    }

    constexpr QuantOption(int mode)
        : mValue(static_cast<Value>(mode))
    {
    }

    constexpr operator Value() const
    {
        return mValue;
    }

    constexpr bool hasPerChannelScaling() const
    {
        return mValue == PerChannelQuant || mValue == PerTokenChannelQuant;
    }

    constexpr bool hasPerTokenScaling() const
    {
        return mValue == PerTokenQuant || mValue == PerTokenChannelQuant;
    }

    constexpr bool hasStaticActivationScaling() const
    {
        return mValue != None && !hasPerTokenScaling();
    }

    constexpr int getQuantMode() const
    {
        return mValue != None ? 2 : 0;
    }

    static constexpr QuantOption make(bool perChannelScaling, bool perTokenScaling)
    {
        if (perChannelScaling && perTokenScaling)
        {
            return PerTokenChannelQuant;
        }
        else if (perChannelScaling)
        {
            return PerChannelQuant;
        }
        else if (perTokenScaling)
        {
            return PerTokenQuant;
        }
        return PerTensorQuant;
    }

private:
    Value mValue;
};

} // namespace common
} // namespace tensorrt_llm
