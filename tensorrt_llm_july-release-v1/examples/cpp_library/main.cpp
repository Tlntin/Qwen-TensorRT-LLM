#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tensorrt_llm_libutils.h"

int main(int argc, char* argv[])
{
    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
        {
            if (severity <= nvinfer1::ILogger::Severity::kERROR)
                std::cerr << "[TensorRT-LLM ERR]: " << msg << std::endl;
            else if (severity == nvinfer1::ILogger::Severity::kWARNING)
                std::cerr << "[TensorRT-LLM WARNING]: " << msg << std::endl;
            else
                std::cout << "[TensorRT-LLM LOG]: " << msg << std::endl;
        }
    };

    TRTLogger* trtLogger = new TRTLogger();

    std::string libname = "libtensorrt_llm_plugin.so";

    /* =============== initLibNvInferPlugins =============== */

    typedef bool (*initLibNvInferPlugins_sig)(void*, const void*);

    auto initLibNvInferPlugins = getTrtLLMFunction<initLibNvInferPlugins_sig>(
        /*libFileSoName=*/libname,
        /*symbol=*/"initLibNvInferPlugins");

    std::cout << std::endl;

    std::string libNamespace = "tensorrt_llm";
    const char* libNamespace_cstr = libNamespace.data();

    bool status1 = initLibNvInferPlugins(trtLogger, libNamespace_cstr);
    std::cout << "Success Status: " << status1 << std::endl << std::endl;

    bool status2 = initLibNvInferPlugins(trtLogger, libNamespace_cstr);
    std::cout << "Success Status: " << status2 << std::endl;

    /* =============== getInferLibVersion =============== */

    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    typedef int32_t (*getInferLibVersion_sig)();

    auto getInferLibVersion = getTrtLLMFunction<getInferLibVersion_sig>(
        /*libFileSoName=*/libname,
        /*symbol=*/"getInferLibVersion");

    std::cout << std::endl;

    int32_t version = getInferLibVersion();
    std::cout << "Version: " << version << std::endl;

    return 0;
}
