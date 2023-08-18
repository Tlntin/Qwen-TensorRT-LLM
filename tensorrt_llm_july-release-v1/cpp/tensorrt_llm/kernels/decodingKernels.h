/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "gptKernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{

namespace kernels
{

void invokeGatherTree(int* beams, int* max_sequence_lengths, const int max_time, const int batch_size,
    const int beam_width, const int* step_ids, const int* parent_ids, const int* end_tokens, cudaStream_t stream);

void invokeGatherTree(int* beams, int* max_sequence_lengths, const int max_time, const int batch_size,
    const int beam_width, const int* step_ids, const int* parent_ids, const int* end_tokens, const int max_input_length,
    cudaStream_t stream);

struct gatherTreeParam
{
    int* beams = nullptr;
    int* max_sequence_lengths = nullptr;
    int max_sequence_length_final_step = 0;
    const int* input_lengths = nullptr;
    // response input lengths (used to slice the ids during postprocessing)
    int* response_input_lengths = nullptr;
    int max_time = 0;
    int batch_size = 0;
    int beam_width = 0;
    const int* step_ids = nullptr;
    const int* parent_ids = nullptr;
    const int* end_tokens = nullptr;
    int max_input_length = 0;
    const int* prefix_soft_prompt_lengths = nullptr;
    // p_prompt_tuning prompt leangths, used to remove prompts during post-processing
    const int* p_prompt_tuning_prompt_lengths = nullptr;
    int max_input_without_prompt_length = 0;
    // prefix soft prompt
    int max_prefix_soft_prompt_length = 0;
    int* output_ids = nullptr;
    // True if we have virtual padding tokens to fill up to max_input_len
    bool has_padding = true;
    cudaStream_t stream;
};

/*
Do gatherTree on beam search to get final result.
*/
void invokeGatherTree(gatherTreeParam param);

} // namespace kernels
} // namespace tensorrt_llm
