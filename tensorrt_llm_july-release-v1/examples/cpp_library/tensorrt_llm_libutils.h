#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

#include "NvInfer.h"

template <typename tSymbolSignature>
tSymbolSignature getTrtLLMFunction(std::string libFileSoName, std::string symbol)
{
    std::cout << "Trying to load " << libFileSoName << " ..." << std::endl;

    // 1. Defining a handle to the library
    void* handle = dlopen(libFileSoName.c_str(), RTLD_LAZY | RTLD_GLOBAL);

    // 2. Check for errors
    const char* dl_error1 = dlerror();
    if (!handle)
    {
        throw std::runtime_error("Cannot open library: " + std::string(dl_error1));
    }

    // 3. Load actual queried `symbol`
    std::cout << "Loading symbol `" << symbol << "` ..." << std::endl;

    tSymbolSignature symbolFctn = nullptr;
    *(void**) (&symbolFctn) = dlsym(handle, symbol.c_str());

    // 4. Check for errors
    const char* dl_error2 = dlerror();
    if (dl_error2)
    {
        dlclose(handle);
        throw std::runtime_error("Cannot load symbol '" + symbol + "': " + std::string(dl_error2));
    }

    return symbolFctn;
}
