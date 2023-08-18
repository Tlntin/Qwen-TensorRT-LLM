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

#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

template <typename T>
__global__ void update_kernel(bool* finished, int* parent_ids, int* sequence_length, int* word_ids, int* output_ids,
    BeamHypotheses beam_hyps, const int vocab_size, const int* end_ids, const int local_batch_size,
    const int beam_width)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < local_batch_size * beam_width;
         index += blockDim.x * gridDim.x)
    {

        int batch_id = index / beam_width;
        sequence_length[index] = finished[index] ? sequence_length[index] : sequence_length[index] + 1;

        int beam_id = (word_ids[index] / vocab_size) % beam_width;
        int word_id = word_ids[index] % vocab_size;

        sequence_length[index] = sequence_length[batch_id * beam_width + beam_id];
        finished[index] = word_id == end_ids[index / beam_width] ? 1 : 0;
        parent_ids[index] = beam_id;
        word_ids[index] = word_id;
        output_ids[index] = word_id;

        if (beam_hyps.num_beams != nullptr)
        {
            if (beam_hyps.num_beams[beam_hyps.ite * beam_hyps.local_batch_size + batch_id] == beam_width)
            {
                for (int i = 0; i < beam_width; i++)
                {
                    finished[batch_id * beam_width + i] = true;
                }
            }
        }
    }
}

void invokeUpdate(bool* finished, int* parent_ids, int* sequence_length, int* word_ids, int* output_ids,
    BeamHypotheses* beam_hyps, const int local_batch_size, const int beam_width, const int vocab_size_padded,
    const int* end_ids, cudaStream_t stream)
{
    dim3 grid((int) ceil(local_batch_size * beam_width * 1.0 / 256));
    dim3 block(256);

    update_kernel<float><<<grid, block, 0, stream>>>(finished, parent_ids, sequence_length, word_ids, output_ids,
        *beam_hyps, vocab_size_padded, end_ids, local_batch_size, beam_width);
}

template <typename T>
void OnlineBeamSearchLayer<T>::setup(SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseBeamSearchLayer<T>::setupBase(setupParams);

    mDiversityRate = setupParams.beam_search_diversity_rate.value_or(0.0f);
    mLengthPenalty = setupParams.length_penalty.value_or(0.0f);
}

template <typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params)
{
    Tensor const& output_ids = outputs.output_ids;
    const int batch_size = output_ids.shape[1];
    const int beam_width = output_ids.shape[2];
    const int step{params.step};
    const int ite{params.ite};
    Tensor const& logits{params.logits};
    const int local_batch_size = logits.shape[0];

    const int id_offset = step * batch_size * beam_width + local_batch_size * ite * beam_width;

    BeamHypotheses beam_hyps;
    auto* const end_ids = params.end_ids.template getPtr<const int>();
    float* output_log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    auto* finished = outputs.finished->template getPtr<bool>();
    auto* sequence_length = outputs.sequence_length->template getPtr<int>();
    if (outputs.beam_hyps)
    {
        beam_hyps = *(reinterpret_cast<BeamHypotheses*>(outputs.beam_hyps.value()));
        beam_hyps.step = step;
        beam_hyps.ite = ite;
        beam_hyps.local_batch_size = local_batch_size;
        beam_hyps.batch_size = output_ids.shape[1];
        beam_hyps.max_seq_len = output_ids.shape[0];
        beam_hyps.output_ids_src = output_ids.getPtr<int>();
        beam_hyps.parent_ids_src = outputs.parent_ids.template getPtr<int>();
        beam_hyps.sequence_lengths_src = sequence_length;
        beam_hyps.log_probs_src = output_log_probs;
        beam_hyps.length_penalty = mLengthPenalty;
        beam_hyps.end_ids = end_ids;
    }

    output_log_probs
        = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtrWithOffset<float>(id_offset) : nullptr;
    invokeTopkSoftMax(logits.template getPtr<T>(), (const T*) (nullptr), finished, sequence_length,
        outputs.cum_log_probs->template getPtr<float>(), output_log_probs, output_ids.getPtrWithOffset<int>(id_offset),
        topk_softmax_workspace_, topk_softmax_workspace_size_, &beam_hyps, local_batch_size, beam_width,
        vocab_size_padded_, end_ids, mDiversityRate, mLengthPenalty, stream_);
    sync_check_cuda_error();

    invokeUpdate(finished, outputs.parent_ids.template getPtrWithOffset<int>(id_offset), sequence_length,
        output_ids.getPtrWithOffset<int>(id_offset), output_ids.getPtrWithOffset<int>(id_offset), &beam_hyps,
        local_batch_size, beam_width, vocab_size_padded_, end_ids, stream_);
    sync_check_cuda_error();
}

template <typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(size_t batch_size, size_t beam_width)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // we need to check 2 * beam_width candidates each time
    // 64 is the max beam width we support now.
    topk_softmax_workspace_size_ = (size_t) (ceil(batch_size * 64 * (64 * 2) / 4.) * 4 * 2
        + ceil(batch_size * (64 * 2) * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * (MAX_K * 2) + 2) / 4.) * 4);

    topk_softmax_workspace_ = reinterpret_cast<float*>(
        allocator_->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, true));
    is_allocate_buffer_ = true;
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    cublasMMWrapper* cublas_wrapper, IAllocator* allocator, bool is_free_buffer_after_forward)
    : BaseBeamSearchLayer<T>(
        vocab_size, vocab_size_padded, stream, cublas_wrapper, allocator, is_free_buffer_after_forward)
{
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer)
    : BaseBeamSearchLayer<T>(beam_search_layer)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
OnlineBeamSearchLayer<T>::~OnlineBeamSearchLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template class OnlineBeamSearchLayer<float>;
template class OnlineBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
