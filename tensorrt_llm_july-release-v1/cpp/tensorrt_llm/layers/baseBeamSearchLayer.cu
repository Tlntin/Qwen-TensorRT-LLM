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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/beamSearchPenaltyKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

__global__ void update_indir_cache_kernel(int* tgt_indir_cache, const int* src_indir_cache, const int* beam_ids,
    const bool* finished, int start_step, int batch_dim, int local_batch_size, int beam_width, int max_seq_len,
    int step)
{
    int time_step = threadIdx.x + blockIdx.x * blockDim.x;
    int bb_id = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch_id = bb_id / beam_width;
    const int beam_id = bb_id % beam_width;

    if (bb_id >= beam_width * local_batch_size || time_step >= min(step + 1, max_seq_len) || finished[bb_id])
    {
        return;
    }
    time_step += start_step;
    const int time_step_circ = time_step % max_seq_len;

    const int src_beam = beam_ids[batch_id * beam_width + beam_id];

    const uint tgt_offset = batch_id * beam_width * max_seq_len + beam_id * max_seq_len + time_step_circ;
    const uint src_offset = batch_id * beam_width * max_seq_len + src_beam * max_seq_len + time_step_circ;

    tgt_indir_cache[tgt_offset] = (time_step == step) ? beam_id : src_indir_cache[src_offset];
}

void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache, const int* beam_ids,
    const bool* finished, int batch_dim, int local_batch_size, int beam_width, int max_seq_len, int step,
    cudaStream_t stream)
{
    const dim3 block(32);
    const int start_step = max(0, step + 1 - max_seq_len);
    const int num_steps = min(step + 1, max_seq_len);
    // Update indirections steps [start_step, step], included
    const dim3 grid((num_steps + block.x - 1) / block.x, local_batch_size * beam_width);
    update_indir_cache_kernel<<<grid, block, 0, stream>>>(tgt_indir_cache, src_indir_cache, beam_ids, finished,
        start_step, batch_dim, local_batch_size, beam_width, max_seq_len, step);
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper, IAllocator* allocator, bool is_free_buffer_after_forward)
    : BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, nullptr)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer)
    : BaseLayer(beam_search_layer)
    , vocab_size_(beam_search_layer.vocab_size_)
    , vocab_size_padded_(beam_search_layer.vocab_size_padded_)
    , topk_softmax_workspace_size_(beam_search_layer.topk_softmax_workspace_size_)
{
}

template <typename T>
BaseBeamSearchLayer<T>::~BaseBeamSearchLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
void BaseBeamSearchLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&topk_softmax_workspace_));
        is_allocate_buffer_ = false;
    }
}

template <typename T>
void BaseBeamSearchLayer<T>::setupBase(SetupParams const& setupParams)
{
    mTemperature = (setupParams.temperature) ? setupParams.temperature->at(0) : 1.0f;
    mMinLength = (setupParams.min_length) ? setupParams.min_length->at(0) : 0;

    mRepetitionPenaltyType = RepetitionPenaltyType::None;
    mRepetitionPenalty = getDefaultPenaltyValue(mRepetitionPenaltyType);
    if (setupParams.repetition_penalty || setupParams.presence_penalty)
    {
        TLLM_CHECK_WITH_INFO(!(setupParams.repetition_penalty && setupParams.presence_penalty),
            "Found ambiguous parameters repetition_penalty and presence_penalty "
            "which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        mRepetitionPenaltyType
            = setupParams.repetition_penalty ? RepetitionPenaltyType::Multiplicative : RepetitionPenaltyType::Additive;
        mRepetitionPenalty = mRepetitionPenaltyType == RepetitionPenaltyType::Multiplicative
            ? setupParams.repetition_penalty->at(0)
            : setupParams.presence_penalty->at(0);
    }
}

template <typename T>
void BaseBeamSearchLayer<T>::forward(BeamSearchOutputParams& outputs, ForwardParams const& params)
{
    Tensor& output_ids = outputs.output_ids;

    const int batch_size = output_ids.shape[1];
    const int beam_width = output_ids.shape[2];
    allocateBuffer(batch_size, beam_width);

    const int step{params.step};
    const int ite{params.ite};
    Tensor const& logits = params.logits;
    const int local_batch_size = logits.shape[0];

    const T* embedding_bias = params.embedding_bias ? params.embedding_bias->template getPtr<const T>() : nullptr;

    auto* end_ids = params.end_ids.template getPtr<const int>();
    auto* const input_lengths = params.input_lengths ? params.input_lengths->template getPtr<const int>() : nullptr;
    int* sequence_length = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    invokeAddBiasApplyPenalties(step, logits.getPtr<T>(),
        output_ids.getPtrWithOffset<const int>(
            (step - 1) * batch_size * beam_width + ite * local_batch_size * beam_width),
        output_ids.getPtr<const int>(), outputs.parent_ids.template getPtr<const int>(), input_lengths, sequence_length,
        embedding_bias, ite, params.max_input_length, local_batch_size, batch_size, beam_width, vocab_size_,
        vocab_size_padded_, end_ids, mTemperature, mRepetitionPenalty, mRepetitionPenaltyType, mMinLength, stream_);
    sync_check_cuda_error();

    invokeSoftMax(outputs, params);

    if (beam_width > 1)
    {
        const int max_seq_len = output_ids.shape[0];

        update_indir_cache_kernelLauncher(outputs.tgt_cache_indirection.template getPtr<int>(),
            params.src_cache_indirection.template getPtr<const int>(),
            outputs.parent_ids.template getPtrWithOffset<const int>(
                +step * beam_width * batch_size + ite * local_batch_size * beam_width),
            outputs.finished->template getPtr<const bool>(), batch_size, local_batch_size, beam_width, max_seq_len,
            step, stream_);
        sync_check_cuda_error();
    }
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_)
    {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseBeamSearchLayer<float>;
template class BaseBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
