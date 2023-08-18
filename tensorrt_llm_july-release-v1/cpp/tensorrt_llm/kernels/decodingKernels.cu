/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{

namespace kernels
{

__global__ void gatherTree(gatherTreeParam param)
{
    //  PREFIX SOFT PROMPT
    //  beam: have six parts
    //      [prompt | input | input_padding | prompt_padding | generated output | padding (use end_token)]
    //  parents: have five parts
    //      [prompt | input | input_padding | prompt_padding | generated output | padding (use 0)]
    //  step_ids: need to remove prompt, input_padding and prompt_padding
    //      the shape is [input_length + requested_output_length, bs, beam_width]
    //      need to transpose to output_ids [bs, beam_width, input_length + requested_output_length]
    //  max_input_length: input + input_padding + prompt_padding

    //  P/PROMPT TUNING
    //  NOTE: input (real ids | prompt virtual ids) have already been preprocessed during embedding lookup, no prompt
    //  templates now beam: [input (real ids | prompt virtual ids) | input_padding | generated output | padding (use
    //  end_token)] parents: [input (real ids | prompt virtual ids) | input_padding | generated output | padding (use
    //  0)] step_ids: need to remove virtual prompt ids in input ids
    //      the shape is [input_length (real input length, prompt length) + requested_output_length, bs, beam_width]
    //      need to transpose to output_ids [bs, beam_width, input_length + requested_output_length]
    //  max_input_length: input (real ids | prompt virtual ids) + input_padding

    const int max_input_length = param.input_lengths == nullptr ? 0 : param.max_input_length;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < param.batch_size * param.beam_width;
         i += gridDim.x * blockDim.x)
    {
        const int batch = i / param.beam_width;
        const int beam = i % param.beam_width;
        const int prompt_len
            = param.prefix_soft_prompt_lengths == nullptr ? 0 : param.prefix_soft_prompt_lengths[batch];
        int input_len = param.input_lengths == nullptr ? 0 : param.input_lengths[i];
        // virtual prompts mean the prompt embedded in input ids (with prompt templates) [p/prompt tuning]
        const int virtual_prompt_length
            = param.p_prompt_tuning_prompt_lengths == nullptr ? 0 : param.p_prompt_tuning_prompt_lengths[batch];
        // real input length (without virtual prompts) [p/prompt tuning]
        input_len -= virtual_prompt_length;

        const int* parent_ids = param.parent_ids;
        const int* step_ids = param.step_ids;

        // TODO(bhsueh) optimize the reduce_max operation for large beam_width
        int max_len = -1;
        bool update_response_input_length = param.response_input_lengths != nullptr;
        // int selected_beam_index = 0;
        for (int j = 0; j < param.beam_width; j++)
        {
            int tmp_len
                = param.max_sequence_lengths[batch * param.beam_width + j] + param.max_sequence_length_final_step - 1;
            param.max_sequence_lengths[batch * param.beam_width + j] = tmp_len;
            // TODO(bhsueh) temporally remove codes about prompting
            // update the response input length
            if (update_response_input_length)
            {
                param.response_input_lengths[batch * param.beam_width + j] = input_len;
            }
            if (tmp_len > max_len)
            {
                max_len = tmp_len;
            }
        }
        const int max_seq_len_b = min(param.max_time, max_len);
        if (max_seq_len_b <= 0)
        {
            continue;
        }

#define GET_IX(time_ix, beam_ix)                                                                                       \
    (param.batch_size * param.beam_width * (time_ix) + param.beam_width * batch + (beam_ix))

        const int padding_offset_and_prompt_offset = param.has_padding ? max_input_length - input_len + prompt_len : 0;
        const int initial_tgt_ix = GET_IX(max_seq_len_b - 1 - padding_offset_and_prompt_offset, beam);
        const int initial_parent_ix = GET_IX(max_seq_len_b - 1, beam);
        param.beams[initial_tgt_ix] = __ldg(step_ids + initial_parent_ix);
        int parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + initial_parent_ix) % param.beam_width;
        bool found_bad = false;

        for (int level = max_seq_len_b - 2; level >= 0; --level)
        {
            if (level < prompt_len || (param.has_padding && level >= input_len && level < max_input_length))
            {
                continue;
            }
            int tgt_level = level >= max_input_length ? level - padding_offset_and_prompt_offset : level - prompt_len;
            const int level_beam_ix = GET_IX(tgt_level, beam);
            const int level_parent_ix = GET_IX(level, parent);
            if (parent < 0 || parent > param.beam_width)
            {
                // param.beams[level_beam_ix] = -1;
                param.beams[level_beam_ix] = param.end_tokens[batch];
                parent = -1;
                found_bad = true;
            }
            else
            {
                param.beams[level_beam_ix] = __ldg(step_ids + level_parent_ix);
                parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + level_parent_ix) % param.beam_width;
            }
        }
        // set the padded part as end_token
        // input_len
        for (int index = max_len - padding_offset_and_prompt_offset;
             index < param.max_time - param.max_prefix_soft_prompt_length; ++index)
        {
            param.beams[GET_IX(index, beam)] = param.end_tokens[batch];
        }

        // Not necessary when using a BeamSearchDecoder, but necessary
        // when a user feeds in possibly broken trajectory (i.e., non-eos
        // entries in a beam following eos entries).
        if (!found_bad)
        {
            bool finished = false;
            // skip the step 0 because it is often the start token
            int start_step = max_input_length == 0 ? 1 : max_input_length;
            for (int time = start_step; time < max_seq_len_b; ++time)
            {
                const int level_beam_ix = GET_IX(time, beam);
                if (finished)
                {
                    param.beams[level_beam_ix] = param.end_tokens[batch];
                }
                else if (param.beams[level_beam_ix] == param.end_tokens[batch])
                {
                    finished = true;
                }
            }
        }
#undef GET_IX

        // transpose on output_ids
        // remove p_prompt tuning virtual tokens (end tokens)
        int actual_output_length = param.max_time - param.max_prefix_soft_prompt_length
            - (param.max_input_length - param.max_input_without_prompt_length);
        if (param.output_ids != nullptr)
        {
            for (int j = 0; j < actual_output_length; j++)
            {
                param.output_ids[i * actual_output_length + j]
                    = param.beams[j * param.batch_size * param.beam_width + i];
            }
        }
    }
}

void invokeGatherTree(int* beams, int* max_sequence_lengths, const int max_time, const int batch_size,
    const int beam_width, const int* step_ids, const int* parent_ids, const int* end_tokens, cudaStream_t stream)
{
    gatherTreeParam param;
    param.beams = beams;
    param.max_sequence_lengths = max_sequence_lengths;
    param.max_time = max_time;
    param.batch_size = batch_size;
    param.beam_width = beam_width;
    param.step_ids = step_ids;
    param.parent_ids = parent_ids;
    param.end_tokens = end_tokens;
    param.max_input_length = 1;
    param.prefix_soft_prompt_lengths = nullptr;
    param.stream = stream;
    invokeGatherTree(param);
}

void invokeGatherTree(int* beams, int* max_sequence_lengths, const int max_time, const int batch_size,
    const int beam_width, const int* step_ids, const int* parent_ids, const int* end_tokens, const int max_input_length,
    cudaStream_t stream)
{
    gatherTreeParam param;
    param.beams = beams;
    param.max_sequence_lengths = max_sequence_lengths;
    param.max_time = max_time;
    param.batch_size = batch_size;
    param.beam_width = beam_width;
    param.step_ids = step_ids;
    param.parent_ids = parent_ids;
    param.end_tokens = end_tokens;
    param.max_input_length = max_input_length;
    param.prefix_soft_prompt_lengths = nullptr;
    param.stream = stream;
    invokeGatherTree(param);
}

void invokeGatherTree(gatherTreeParam param)
{
    int batchbeam = param.batch_size * param.beam_width;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024)
    {
        grid.x = ceil(param.batch_size * param.beam_width / 1024.);
        block.x = 1024;
    }
    gatherTree<<<grid, block, 0, param.stream>>>(param);
}

} // namespace kernels
} // namespace tensorrt_llm
