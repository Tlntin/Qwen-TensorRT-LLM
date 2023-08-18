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

#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;

namespace torch_ext
{

// this should be similar to gatherTree in cpp/tensorrt_llm/runtime/gptSession.cpp
th::Tensor gatherTree(th::Tensor& sequence_lengths, th::Tensor& output_ids, th::Tensor& parent_ids, th::Tensor& end_ids,
    th::Tensor& tiled_input_lengths, int64_t batch_size, int64_t beam_width, int64_t max_input_length,
    int64_t max_session_len)
{
    th::Tensor workspace = torch::zeros(batch_size * beam_width * max_session_len * sizeof(int32_t),
        torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    th::Tensor final_output_ids = torch::zeros({batch_size, beam_width, max_session_len},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // For sampling, it is equivalent to all parent ids are 0.
    tl::kernels::gatherTreeParam param;
    param.beams = get_ptr<int32_t>(workspace);
    // Remove prompt length if possible
    param.max_sequence_lengths = get_ptr<int32_t>(sequence_lengths);
    // add sequence_length 1 here because the sequence_length of time step t is t - 1
    param.max_sequence_length_final_step = 1;
    // response input lengths (used to slice the ids during postprocessing), used in interactive generation
    // This feature is not supported yet, setting it to nullptr temporarily.
    param.response_input_lengths = nullptr;
    param.max_time = max_session_len;
    param.batch_size = batch_size;
    param.beam_width = beam_width;
    param.step_ids = get_ptr<int32_t>(output_ids);
    param.parent_ids = beam_width == 1 ? nullptr : get_ptr<int32_t>(parent_ids);
    param.end_tokens = get_ptr<int32_t>(end_ids);
    param.max_input_length = max_input_length;
    param.input_lengths = get_ptr<int32_t>(tiled_input_lengths);

    // no prompt supporting now
    // This feature is not supported yet, setting it to nullptr temporarily.
    param.prefix_soft_prompt_lengths = nullptr;
    param.p_prompt_tuning_prompt_lengths = nullptr;
    param.max_input_without_prompt_length = max_input_length;
    param.max_prefix_soft_prompt_length = 0;

    param.stream = at::cuda::getCurrentCUDAStream().stream();
    param.output_ids = get_ptr<int32_t>(final_output_ids);
    // NOTE: need to remove all prompt virtual tokens

    tl::kernels::invokeGatherTree(param);
    sync_check_cuda_error();
    return final_output_ids;
}

} // namespace torch_ext

static auto gather_tree = torch::RegisterOperators("tensorrt_llm::gather_tree", &torch_ext::gatherTree);
