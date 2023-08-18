#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace trt = nvinfer1;

namespace
{
void benchmarkGptSession(std::string const& modelName, std::filesystem::path const& dataPath, bool inputPacked,
    std::vector<int> const& batchSizes, std::vector<std::vector<int>> const& inOutLen,
    std::shared_ptr<nvinfer1::ILogger> const& logger)
{
    auto const json = GptJsonConfig::parse(dataPath / "config.json");
    auto const modelConfig = json.modelConfig();
    auto const worldConfig = WorldConfig::mpi(*logger);
    auto const enginePath = dataPath / json.engineFilename(worldConfig, modelName);
    auto const dtype = modelConfig.getDataType();
    auto const useHalf = (dtype == nvinfer1::DataType::kHALF);

    auto constexpr beamWidth = 1;
    SamplingConfig samplingConfig{beamWidth};
    samplingConfig.temperature = std::vector{1.0f};
    samplingConfig.minLength = std::vector{1};
    samplingConfig.randomSeed = std::vector{42ull};
    samplingConfig.topK = std::vector{1};
    samplingConfig.topP = std::vector{0.0f};

    GptSession session{modelConfig, worldConfig, enginePath, logger};
    // Use bufferManager for copying data to and from the GPU
    auto& bufferManager = session.getBufferManager();

    for (auto inOut : inOutLen)
    {
        auto const maxInputLength = inOut[0];
        auto const maxNewTokens = inOut[1];

        auto constexpr endId = 50256;
        auto constexpr padId = 50256;

        for (auto const batchSize : batchSizes)
        {
            session.setup(batchSize, maxInputLength, maxNewTokens, samplingConfig);

            std::vector<SizeType> inputLenghtsHost(batchSize, maxInputLength);
            auto inputLenghts
                = bufferManager.copyFrom(inputLenghtsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

            // copy inputs and wrap into shared_ptr
            GenerationInput::TensorPtr inputIds;
            std::vector<std::int32_t> inputsHost(batchSize * maxInputLength, padId);
            inputIds
                = bufferManager.copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);

            GenerationInput generationInput{endId, padId, std::move(inputIds), std::move(inputLenghts), inputPacked};
            generationInput.disableInputCopy = true;

            // runtime will allocate memory for output if this tensor is empty
            GenerationOutput generationOutput{bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32)};

            auto constexpr warmUps = 2;
            for (auto r = 0; r < warmUps; ++r)
            {
                SizeType numSteps = 0;
                generationOutput.onTokenGenerated
                    = [&numSteps, maxNewTokens](
                          GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished) { ++numSteps; };
                session.generate(generationOutput, generationInput);
                bufferManager.getStream().synchronize();
            }

            // repeat the same inputs multiple times
            cudaDeviceSynchronize();
            auto const start = std::chrono::steady_clock::now();

            auto constexpr repetitions = 10;
            for (auto r = 0; r < repetitions; ++r)
            {
                SizeType numSteps = 0;
                generationOutput.onTokenGenerated
                    = [&numSteps, maxNewTokens](
                          GenerationOutput::TensorPtr const& outputIds, SizeType step, bool finished) { ++numSteps; };
                session.generate(generationOutput, generationInput);
                bufferManager.getStream().synchronize();
            }

            auto const end = std::chrono::steady_clock::now();
            auto averageLatency
                = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count())
                / (repetitions * 1000);
            if (worldConfig.getRank() == 0)
            {
                printf("[BENCHMARK] request_batch_size %d inlen %d outlen %d latency(ms) %.2f\n", batchSize,
                    maxInputLength, maxNewTokens, averageLatency);
            }
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cout << "Usage: " << argv[0] << " model_name engine_dir batch_size in_out_len" << std::endl;
        std::cout << "model_name -- Model name specified for engines." << std::endl;
        std::cout << "engine_dir -- Directory that store the engines." << std::endl;
        std::cout << "batch_size -- Specify batch size(s) you want to benchmark. Multiple batch "
                  << "sizes can be separated by \";\", example: \"1;8;64\"." << std::endl;
        std::cout << "in_out_len -- Specify input-output length(s) you want to benchmark, this "
                  << "option is mainly for GPT and GPT-like models. Multiple input "
                  << "lengths can be separated by \";\", example: \"60,20;128,20\"." << std::endl;
        return 0;
    }

    // Argument 1: Model name
    std::string const modelName = std::string(argv[1]);

    // Argument 2: Engine directory
    std::string const dataPath = std::string(argv[2]);

    // Argument 3: Batch sizes
    std::istringstream ssBatchSizesArg;
    ssBatchSizesArg.str(std::string(argv[3]));
    std::vector<int> batchSizes;
    for (std::string token; std::getline(ssBatchSizesArg, token, ';');)
    {
        batchSizes.push_back(std::stoi(token));
    }

    // Argument 4: Input-output lengths
    std::istringstream ssInOutLenArg;
    ssInOutLenArg.str(std::string(argv[4]));
    std::vector<std::vector<int>> inOutLen;
    for (std::string token; std::getline(ssInOutLenArg, token, ';');)
    {
        std::istringstream ssTmp(token);
        std::vector<int> inOut;
        for (std::string t; std::getline(ssTmp, t, ',');)
        {
            inOut.push_back(std::stoi(t));
        }
        inOutLen.push_back(inOut);
    }

    bool constexpr inputPacked = false;
    auto logger = std::make_shared<TllmLogger>();
    logger->setLevel(trt::ILogger::Severity::kERROR);
    initLibNvInferPlugins(logger.get(), "tensorrt_llm");
    benchmarkGptSession(modelName, dataPath, inputPacked, batchSizes, inOutLen, logger);
    return 0;
}
