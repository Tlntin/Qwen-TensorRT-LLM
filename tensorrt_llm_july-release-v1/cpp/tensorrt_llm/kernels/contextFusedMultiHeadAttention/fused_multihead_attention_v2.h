/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "cuda_runtime_api.h"
#include "fused_multihead_attention_common.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tmaDescriptor.h"
#include <assert.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

// compute groups for warp-specialized kernels on Hopper
#define NUM_COMPUTE_GROUPS 2

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params_v2
{
    // The QKV matrices.
    void* qkv_ptr;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use trick to avoid I2F/F2I in the INT8 kernel.
    bool enable_i2f_trick;

    // array of lengths b+1 holding prefix sum of actual sequence lengths
    int* cu_seqlens;

    // use C/32 Format.
    bool interleaved = false;
    bool use_int8_scale_max = false;

    // only have one head for keys/values
    bool multi_query_attention = false;

    // is input/output padded
    bool is_s_padded = false;

    // tma descriptors
    cudaTmaDesc tma_desc_q;
    cudaTmaDesc tma_desc_k;
    cudaTmaDesc tma_desc_v;

    void clear()
    {
        qkv_ptr = nullptr;
        packed_mask_ptr = nullptr;
        o_ptr = nullptr;

        qkv_stride_in_bytes = 0;
        packed_mask_stride_in_bytes = 0;
        o_stride_in_bytes = 0;
#if defined(STORE_P)
        p_ptr = nullptr;
        p_stride_in_bytes = 0
#endif // defined(STORE_P)

#if defined(STORE_S)
            s_ptr
            = nullptr;
        s_stride_in_bytes = 0;
#endif // defined(STORE_S)

        b = 0;
        h = 0;
        s = 0;
        d = 0;
        // The scaling factors for the kernel.
        scale_bmm1 = 0;
        scale_softmax = 0;
        scale_bmm2 = 0;

        enable_i2f_trick = false;

        cu_seqlens = nullptr;
        interleaved = false;
        use_int8_scale_max = false;

        multi_query_attention = false;

        is_s_padded = false;
    }
};

// flags to control kernel choice
struct Launch_params
{
    // seq_length to select the kernel
    int kernel_s = 0;
    // flags to control small batch kernel choice
    // true: never unroll
    bool ignore_b1opt = false;
    // true: always unroll
    bool force_unroll = false;
    // use fp32 accumulation
    bool force_fp32_acc = false;
    // the C/32 format
    bool interleaved = false;
    // by default TMA is not used.
    bool use_tma = false;
    // host seqlens to set tma descriptors
    int* seqlens = nullptr;
    // if flash attention is used (only FP16)
    bool flash_attention = false;
    // if warp_specialized kernels are used (only SM90 HGMMA + TMA)
    bool warp_specialization = false;
    // granular tiling flash attention kernels
    bool granular_tiling = false;
    // mask
    bool causal_mask = false;
    // harward properties to determine how to launch blocks
    int multi_processor_count = 0;
    int device_l2_cache_size = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

extern unsigned char cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin[];
extern unsigned char cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin[];

extern uint32_t cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin_len;
extern uint32_t cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin_len;

////////////////////////////////////////////////////////////////////////////////////////////////////

// S = 0 denotes that any sequence length is supported (flash attention)
#define S 0

static const struct FusedMultiHeadAttentionKernelMetaInfoV2
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
    unsigned int mUnrollStep;
    bool mInterleaved;
    bool mFlashAttention;
    bool mForceFP32Acc;
    bool mCausalMask;
    bool mTiled;
} sMhaKernelMetaInfosV2[] = {
#if CUDA_VERSION >= 11000
    // ampere
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_sm80_kernel_nl", 8192, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm80_kernel_nl", 8192, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_sm80_kernel_nl", 16384, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm80_kernel_nl", 16384, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm80_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm80_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm80_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm80_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm80_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm80_kernel_nl_tiled", 20480, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm80_kernel_nl_tiled", 40960, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm80_kernel_nl", 8192, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_causal_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_causal_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_sm80_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_causal_sm80_kernel_nl", 8192, 128, 64, false, true, true, true, false},
    {DATA_TYPE_BF16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_sm80_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_sm86_kernel_nl_tiled", 20480, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm86_kernel_nl_tiled", 20480, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_sm86_kernel_nl_tiled", 40960, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm86_kernel_nl_tiled", 40960, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_sm86_kernel_nl", 8192, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm86_kernel_nl", 8192, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_sm86_kernel_nl", 16384, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm86_kernel_nl", 16384, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_sm86_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm86_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_sm86_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm86_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_sm86_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm86_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_sm86_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm86_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_sm86_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm86_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_sm86_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm86_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_BF16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_sm86_kernel_nl_tiled", 20480, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_causal_sm86_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_sm86_kernel_nl_tiled", 40960, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_causal_sm86_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_sm86_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_causal_sm86_kernel_nl", 8192, 128, 64, false, true, true, true, false},
    {DATA_TYPE_BF16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_sm86_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_causal_sm86_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_sm86_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_causal_sm86_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_sm86_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_causal_sm86_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_sm86_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_causal_sm86_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_sm86_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_causal_sm86_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_sm86_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_causal_sm86_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_sm86_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_causal_sm86_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm80_kernel_nl_tiled", 20480, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm80_kernel_nl_tiled", 40960, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_causal_sm80_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm80_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm80_kernel_nl", 8192, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm80_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm80_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_kernel_nl", 49152, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm80_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_80, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm80_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm80_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm86_kernel_nl_tiled", 20480, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm86_kernel_nl_tiled", 40960, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_causal_sm86_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm86_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm86_kernel_nl", 8192, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm86_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm86_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm86_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm86_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_kernel_nl", 49152, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm86_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm86_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_86, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm86_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm86_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
#endif

#if CUDA_VERSION >= 11080
    // ada
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_sm89_kernel_nl_tiled", 20480, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_16_causal_sm89_kernel_nl_tiled", 20480, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_sm89_kernel_nl_tiled", 40960, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_32_causal_sm89_kernel_nl_tiled", 40960, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_40_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_128_S_64_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_80_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_128_64_S_128_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_160_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, false, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_256_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, false,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_sm89_kernel_nl", 8192, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_16_causal_sm89_kernel_nl", 8192, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_sm89_kernel_nl", 16384, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_32_causal_sm89_kernel_nl", 16384, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_sm89_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_40_causal_sm89_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_sm89_kernel_nl", 24576, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_64_causal_sm89_kernel_nl", 24576, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_sm89_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_80_causal_sm89_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_sm89_kernel_nl", 49152, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_32_S_128_causal_sm89_kernel_nl", 49152, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_sm89_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_160_causal_sm89_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_sm89_kernel_nl", 98304, 128, 64, false, true, false, false, false},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_16_S_256_causal_sm89_kernel_nl", 98304, 128, 64, false, true, false, true,
        false},
    {DATA_TYPE_BF16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_sm89_kernel_nl_tiled", 20480, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_16_causal_sm89_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_sm89_kernel_nl_tiled", 40960, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_32_causal_sm89_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_40_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_128_S_64_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_80_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_128_64_S_128_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_160_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_BF16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_256_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_BF16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_sm89_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_16_causal_sm89_kernel_nl", 8192, 128, 64, false, true, true, true, false},
    {DATA_TYPE_BF16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_sm89_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_32_causal_sm89_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_sm89_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_40_causal_sm89_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_sm89_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_64_causal_sm89_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_sm89_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_80_causal_sm89_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_sm89_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_32_S_128_causal_sm89_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_sm89_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_160_causal_sm89_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_sm89_kernel_nl", 98304, 128, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_16_S_256_causal_sm89_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_kernel_nl_tiled", 20480, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_16_causal_sm89_kernel_nl_tiled", 20480, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_kernel_nl_tiled", 40960, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_32_causal_sm89_kernel_nl_tiled", 40960, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_40_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_128_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_128_S_64_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_80_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_kernel_nl_tiled", 81920, 128, 128, false, true, true,
        false, true},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_128_64_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_128_64_S_128_causal_sm89_kernel_nl_tiled", 81920, 128, 128, false, true,
        true, true, true},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_160_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true, false,
        true},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_256_causal_sm89_kernel_nl_tiled", 81920, 128, 64, false, true, true,
        true, true},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_kernel_nl", 8192, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 16, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_16_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_16_causal_sm89_kernel_nl", 8192, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_kernel_nl", 16384, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 32, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_32_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_32_causal_sm89_kernel_nl", 16384, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 40, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_40_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_40_causal_sm89_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_kernel_nl", 24576, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 64, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_64_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_64_causal_sm89_kernel_nl", 24576, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_kernel_nl", 49152, 128, 64, false, true, true, false, false},
    {DATA_TYPE_FP16, S, 80, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_80_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_80_causal_sm89_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_kernel_nl", 49152, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_32_S_128_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_32_S_128_causal_sm89_kernel_nl", 49152, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 160, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_160_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_160_causal_sm89_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_kernel_nl", 98304, 128, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_89, cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_16_S_256_sm89_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_16_S_256_causal_sm89_kernel_nl", 98304, 128, 64, false, true, true, true,
        false},
    // hopper
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_32_ldgsts_sm90_kernel", 17408, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_32_causal_ldgsts_sm90_kernel", 17408, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_32_ldgsts_sm90_kernel_nl", 17408, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_32_causal_ldgsts_sm90_kernel_nl", 17408,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_32_ldgsts_sm90_kernel", 25600, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_32_causal_ldgsts_sm90_kernel", 25600, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_32_ldgsts_sm90_kernel_nl", 25600, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_32_causal_ldgsts_sm90_kernel_nl", 25600,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_32_ldgsts_sm90_kernel", 41984, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_32_causal_ldgsts_sm90_kernel", 41984, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_32_ldgsts_sm90_kernel_nl", 41984, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_32_causal_ldgsts_sm90_kernel_nl", 41984,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_ldgsts_sm90_kernel", 33792, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_causal_ldgsts_sm90_kernel", 33792, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_ldgsts_sm90_kernel_nl", 33792, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_64_64_causal_ldgsts_sm90_kernel_nl", 33792,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_ldgsts_sm90_kernel", 50176, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_causal_ldgsts_sm90_kernel", 50176, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_ldgsts_sm90_kernel_nl", 50176, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_128_64_causal_ldgsts_sm90_kernel_nl", 50176,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_ldgsts_sm90_kernel", 82944, 128, 0,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_causal_ldgsts_sm90_kernel", 82944, 128,
        0, false, false, false, true, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_ldgsts_sm90_kernel_nl", 82944, 128, 64,
        false, false, false, false, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_256_64_causal_ldgsts_sm90_kernel_nl", 82944,
        128, 64, false, false, false, true, false},
    {DATA_TYPE_BF16, 64, 32, kSM_90, cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_32_ldgsts_sm90_kernel", 17408, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 64, 32, kSM_90, cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_32_causal_ldgsts_sm90_kernel", 17408, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 64, 32, kSM_90, cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_32_ldgsts_sm90_kernel_nl", 17408, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 64, 32, kSM_90, cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_32_causal_ldgsts_sm90_kernel_nl", 17408,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_BF16, 128, 32, kSM_90, cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_32_ldgsts_sm90_kernel", 25600, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 128, 32, kSM_90, cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_32_causal_ldgsts_sm90_kernel", 25600, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 128, 32, kSM_90, cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_32_ldgsts_sm90_kernel_nl", 25600, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 128, 32, kSM_90, cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_32_causal_ldgsts_sm90_kernel_nl", 25600,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_BF16, 256, 32, kSM_90, cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_32_ldgsts_sm90_kernel", 41984, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 256, 32, kSM_90, cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_32_causal_ldgsts_sm90_kernel", 41984, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 256, 32, kSM_90, cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_32_ldgsts_sm90_kernel_nl", 41984, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 256, 32, kSM_90, cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_32_causal_ldgsts_sm90_kernel_nl", 41984,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_BF16, 64, 64, kSM_90, cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_64_ldgsts_sm90_kernel", 33792, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 64, 64, kSM_90, cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_64_causal_ldgsts_sm90_kernel", 33792, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 64, 64, kSM_90, cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_64_ldgsts_sm90_kernel_nl", 33792, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 64, 64, kSM_90, cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_64_64_causal_ldgsts_sm90_kernel_nl", 33792,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_BF16, 128, 64, kSM_90, cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_64_ldgsts_sm90_kernel", 50176, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 128, 64, kSM_90, cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_64_causal_ldgsts_sm90_kernel", 50176, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 128, 64, kSM_90, cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_64_ldgsts_sm90_kernel_nl", 50176, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 128, 64, kSM_90, cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_128_64_causal_ldgsts_sm90_kernel_nl", 50176,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_BF16, 256, 64, kSM_90, cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_64_ldgsts_sm90_kernel", 82944, 128, 0,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 256, 64, kSM_90, cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_64_causal_ldgsts_sm90_kernel", 82944, 128,
        0, false, false, true, true, false},
    {DATA_TYPE_BF16, 256, 64, kSM_90, cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_64_ldgsts_sm90_kernel_nl", 82944, 128, 64,
        false, false, true, false, false},
    {DATA_TYPE_BF16, 256, 64, kSM_90, cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_bf16_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_bf16_256_64_causal_ldgsts_sm90_kernel_nl", 82944,
        128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_32_ldgsts_sm90_kernel", 17408,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_32_causal_ldgsts_sm90_kernel",
        17408, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_32_ldgsts_sm90_kernel_nl", 17408,
        128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 64, 32, kSM_90, cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_32_causal_ldgsts_sm90_kernel_nl",
        17408, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_32_ldgsts_sm90_kernel", 25600,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_32_causal_ldgsts_sm90_kernel",
        25600, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_32_ldgsts_sm90_kernel_nl",
        25600, 128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 128, 32, kSM_90, cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_32_ldgsts_sm90_cu_cubin_len,
        "fmha_v2_fp16_fp32_128_32_causal_ldgsts_sm90_kernel_nl", 25600, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_32_ldgsts_sm90_kernel", 41984,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_32_causal_ldgsts_sm90_kernel",
        41984, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_32_ldgsts_sm90_kernel_nl",
        41984, 128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 256, 32, kSM_90, cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_32_ldgsts_sm90_cu_cubin_len,
        "fmha_v2_fp16_fp32_256_32_causal_ldgsts_sm90_kernel_nl", 41984, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_64_ldgsts_sm90_kernel", 33792,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_64_causal_ldgsts_sm90_kernel",
        33792, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_64_ldgsts_sm90_kernel_nl", 33792,
        128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 64, 64, kSM_90, cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_64_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_64_64_causal_ldgsts_sm90_kernel_nl",
        33792, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel", 50176,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel",
        50176, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_128_64_ldgsts_sm90_kernel_nl",
        50176, 128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 128, 64, kSM_90, cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_128_64_ldgsts_sm90_cu_cubin_len,
        "fmha_v2_fp16_fp32_128_64_causal_ldgsts_sm90_kernel_nl", 50176, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_64_ldgsts_sm90_kernel", 82944,
        128, 0, false, false, true, false, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_64_causal_ldgsts_sm90_kernel",
        82944, 128, 0, false, false, true, true, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin_len, "fmha_v2_fp16_fp32_256_64_ldgsts_sm90_kernel_nl",
        82944, 128, 64, false, false, true, false, false},
    {DATA_TYPE_FP16, 256, 64, kSM_90, cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin,
        cubin_fmha_v2_fp16_fp32_256_64_ldgsts_sm90_cu_cubin_len,
        "fmha_v2_fp16_fp32_256_64_causal_ldgsts_sm90_kernel_nl", 82944, 128, 64, false, false, true, true, false},
    {DATA_TYPE_FP16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_kernel", 73920, 384, 64, false, true, false, false,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_256_S_32_causal_tma_ws_sm90_kernel", 73920, 384, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_kernel", 147648, 384, 64, false, true, false, false,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_256_S_64_causal_tma_ws_sm90_kernel", 147648, 384, 64, false, true, false, true,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_kernel", 164032, 384, 64, false, true, false, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_128_S_128_causal_tma_ws_sm90_kernel", 164032, 384, 64, false, true, false,
        true, false},
    {DATA_TYPE_FP16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_kernel", 196800, 384, 64, false, true, false, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_64_64_S_256_causal_tma_ws_sm90_kernel", 196800, 384, 64, false, true, false, true,
        false},
    {DATA_TYPE_BF16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_kernel", 73920, 384, 64, false, true, true, false, false},
    {DATA_TYPE_BF16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_256_S_32_causal_tma_ws_sm90_kernel", 73920, 384, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_kernel", 147648, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_BF16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_256_S_64_causal_tma_ws_sm90_kernel", 147648, 384, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_kernel", 164032, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_BF16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_128_S_128_causal_tma_ws_sm90_kernel", 164032, 384, 64, false, true, true, true,
        false},
    {DATA_TYPE_BF16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_kernel", 196800, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_BF16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_bf16_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_bf16_64_64_S_256_causal_tma_ws_sm90_kernel", 196800, 384, 64, false, true, true, true,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_kernel", 73920, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 32, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_32_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_256_S_32_causal_tma_ws_sm90_kernel", 73920, 384, 64, false, true, true,
        true, false},
    {DATA_TYPE_FP16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_kernel", 147648, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 64, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_256_S_64_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_256_S_64_causal_tma_ws_sm90_kernel", 147648, 384, 64, false, true, true,
        true, false},
    {DATA_TYPE_FP16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_kernel", 164032, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 128, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_128_S_128_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_128_S_128_causal_tma_ws_sm90_kernel", 164032, 384, 64, false, true, true,
        true, false},
    {DATA_TYPE_FP16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_kernel", 196800, 384, 64, false, true, true, false,
        false},
    {DATA_TYPE_FP16, S, 256, kSM_90, cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin,
        cubin_fmha_v2_flash_attention_fp16_fp32_64_64_S_256_tma_ws_sm90_cu_cubin_len,
        "fmha_v2_flash_attention_fp16_fp32_64_64_S_256_causal_tma_ws_sm90_kernel", 196800, 384, 64, false, true, true,
        true, false},
#endif
};

#undef S

// meta info for tma warp-specialized kernels
static const struct TmaKernelMetaInfo
{
    unsigned int mD;
    unsigned int mQStep;
    unsigned int mKvStep;
} sTmaMetaInfo[] = {{32, 64, 256}, {64, 64, 256}, {128, 64, 128}, {256, 64, 64}};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Base Class

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    inline uint64_t hashID(unsigned int s, unsigned int d) const
    {
        return (uint64_t) s << 32 | d;
    }

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {
        return hashID(kernelMeta.mS, kernelMeta.mD);
    }

    TFusedMultiHeadAttentionXMMAKernel(
        const TKernelMeta* pMetaStart, unsigned int nMetaCount, Data_type type, unsigned int sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
    }

    void loadXMMAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            const auto& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM == mSM && kernelMeta.mDataType == mDataType)
            {
                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
                int s = static_cast<int>(kernelMeta.mS);
                if (mValidSequences.find(s) == mValidSequences.end())
                    mValidSequences.insert(s);
            }
        }
    }

    bool isValid(int s) const
    {
        return (mValidSequences.find(s) != mValidSequences.end());
    }

    virtual void run(TKernelParam& params, Launch_params& launch_params, cudaStream_t ss) const
    {
        const auto findIter = mFunctions.find(hashID(params.s, params.d));

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                       kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
            mDriver);
    }

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    tensorrt_llm::common::CUDADriverWrapper mDriver;

    Data_type mDataType;
    const TKernelMeta* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<const unsigned char*, CUmodule> mModules;

    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    const TFusedMHAKernelList* getXMMAKernels(const typename TFusedMHAKernelList::KernelMeta* pKernelList,
        unsigned int nbKernels, Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            TFusedMHAKernelList* newKernel = new TFusedMHAKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadXMMAKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TFusedMHAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get()
    {
        int device_id;
        cudaGetDevice(&device_id);
        static std::unique_ptr<TFusedMHAKernelFactory<TFusedMHAKernelList>> s_factory[32] = {nullptr};
        if (s_factory[device_id] == nullptr)
        {
            assert(device_id <= 32);
            s_factory[device_id] = std::make_unique<TFusedMHAKernelFactory<TFusedMHAKernelList>>(
                TFusedMHAKernelFactory<TFusedMHAKernelList>());
        }

        return *(s_factory[device_id]);
    }

private:
    TFusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type type, unsigned int sm) const
    {
        return (uint64_t) type << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
          Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(const FusedMultiHeadAttentionKernelMetaInfoV2* pMetaStart,
        unsigned int nMetaCount, Data_type type, unsigned int sm)
        : TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
            Fused_multihead_attention_params_v2>(pMetaStart, nMetaCount, type, sm)
    {
    }

    inline uint64_t hashID(unsigned int s, unsigned int d, bool interleaved, bool unroll, bool force_fp32_acc,
        bool flash_attention, bool causal_mask, bool tiled) const
    {
        s = flash_attention ? 0 : s;
        // D <= 2048
        return (uint64_t) s << 32 | d << 16 | (tiled ? 32ull : 0ull) | (force_fp32_acc ? 16ull : 0ull)
            | (flash_attention ? 8ull : 0ull) | (causal_mask ? 4ull : 0ull) | (interleaved ? 2ull : 0ull)
            | (unroll ? 1ull : 0ull);
    }

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const
    {

        return hashID(kernelMeta.mS, kernelMeta.mD, kernelMeta.mInterleaved, kernelMeta.mUnrollStep,
            kernelMeta.mForceFP32Acc, kernelMeta.mFlashAttention, kernelMeta.mCausalMask, kernelMeta.mTiled);
    }

    virtual void run(
        Fused_multihead_attention_params_v2& params, Launch_params& launch_params, cudaStream_t stream) const
    {

        bool forceUnroll = launch_params.force_unroll;
        if (!forceUnroll && !launch_params.ignore_b1opt && mSM >= kSM_80)
        {
            const struct
            {
                unsigned int mSM;
                Data_type mDataType;
                int mS;
                int mD;
                int mMaxBatchHead;
            } unrollList[] = {
#if CUDA_VERSION >= 11080
                {kSM_90, DATA_TYPE_FP16, 64, 64, 256},
                {kSM_90, DATA_TYPE_FP16, 128, 64, 128},
                {kSM_90, DATA_TYPE_FP16, 256, 64, 128},
                {kSM_90, DATA_TYPE_FP16, 384, 64, 64},
                {kSM_90, DATA_TYPE_FP16, 512, 64, 64}
#endif
            };
            for (unsigned int i = 0u; i < sizeof(unrollList) / sizeof(unrollList[0]); ++i)
            {
                if (mSM == unrollList[i].mSM && mDataType == unrollList[i].mDataType
                    && launch_params.kernel_s == unrollList[i].mS && params.d == unrollList[i].mD
                    && params.b * params.h <= unrollList[i].mMaxBatchHead)
                {
                    forceUnroll = true;
                    break;
                }
            }
        }

        const auto findIter = mFunctions.find(
            hashID(launch_params.kernel_s, params.d, params.interleaved, forceUnroll, launch_params.force_fp32_acc,
                launch_params.flash_attention, launch_params.causal_mask, launch_params.granular_tiling));
        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "FMHA kernels are not found");

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
        void* kernelParams[] = {&params, nullptr};

        if (!forceUnroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                mDriver);
        } // forceunroll = true for flash attention kernels
        else if (mSM == kSM_90 && launch_params.flash_attention)
        {
            // tricks for launching warp-specialized flash attention kernels on Hopper
            dim3 block_size(1, std::min(params.b * params.h, launch_params.multi_processor_count));

            // distribute m steps to multiple blocks (fully utilize SMs)
            // block.x = blocks that handle single head, block.y = blocks that handle different heads
            size_t sms_per_head = (launch_params.multi_processor_count) / block_size.y;
            size_t m_steps = size_t((params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep);
            m_steps = size_t((m_steps + NUM_COMPUTE_GROUPS - 1) / NUM_COMPUTE_GROUPS) * NUM_COMPUTE_GROUPS;

            size_t size_in_bytes = params.b * params.s * params.qkv_stride_in_bytes;
            if (!launch_params.causal_mask && size_in_bytes <= launch_params.device_l2_cache_size)
            {
                // strategy 1: limit to only 1 wave
                block_size.x = std::min(m_steps / NUM_COMPUTE_GROUPS, sms_per_head);
            }
            else
            {
                // strategy 2: fully unroll the q loops (contiguous blocks handle all q loops)
                block_size.x = m_steps / NUM_COMPUTE_GROUPS;
            }

            cuErrCheck(mDriver.cuLaunchKernel(func, block_size.x, block_size.y, block_size.z, kernelMeta.mThreadsPerCTA,
                           1, 1, kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                mDriver);
        }
        else
        { // forceunroll = true for flash attention kernels
            int unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            TLLM_CHECK_WITH_INFO(kernelMeta.mS == kernelMeta.mUnrollStep * unroll, "Wrong launching sequence length");
            // flash attention supports any sequence length, so we runtime s here
            if (launch_params.flash_attention)
            {
                unroll = (params.s + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep;
            }
            // on Hopper, we still launch blocks (h, b, steps)
            if (mSM == kSM_90)
            {
                cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                               kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                    mDriver);
            } // on Ampere/Ada flash attention, we launch blocks (steps, h, b)
            else
            {
                cuErrCheck(mDriver.cuLaunchKernel(func, unroll, params.h, params.b, kernelMeta.mThreadsPerCTA, 1, 1,
                               kernelMeta.mSharedMemBytes, stream, kernelParams, nullptr),
                    mDriver);
            }
        }
    }
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

inline const FusedMultiHeadAttentionXMMAKernelV2* getXMMAKernelsV2(Data_type type, unsigned int sm)
{
    return FusedMHAKernelFactoryV2::Get().getXMMAKernels(
        sMhaKernelMetaInfosV2, sizeof(sMhaKernelMetaInfosV2) / sizeof(sMhaKernelMetaInfosV2[0]), type, sm);
}

} // namespace kernels
} // namespace tensorrt_llm
