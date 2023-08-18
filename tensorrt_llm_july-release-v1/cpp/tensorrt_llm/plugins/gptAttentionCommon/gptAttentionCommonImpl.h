
#ifndef TRT_GPT_ATTENTION_COMMON_IMPL_H
#define TRT_GPT_ATTENTION_COMMON_IMPL_H

#include "gptAttentionCommon.h"

namespace nvinfer1::plugin
{
template <typename T>
T* GPTAttentionPluginCommon::cloneImpl() const noexcept
{
    static_assert(std::is_base_of_v<GPTAttentionPluginCommon, T>);
    auto* plugin = new T(static_cast<T const&>(*this));
    plugin->setPluginNamespace(mNamespace.c_str());

    // Cloned plugins should be in initialized state with correct resources ready to be enqueued.
    plugin->initialize();
    return plugin;
}

template <typename T>
T* GPTAttentionPluginCreatorCommon::deserializePluginImpl(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GPTAttentionPluginCommon::destroy()
    try
    {
        auto* obj = new T(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace nvinfer1::plugin

#endif // TRT_GPT_ATTENTION_COMMON_IMPL_H
