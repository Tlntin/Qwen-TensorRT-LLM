#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

namespace cutlass
{
namespace gemm
{
namespace kernel
{

template <typename arch>
struct Int8GemmArchTraits
{
    using OperatorClass = cutlass::arch::OpClassSimt;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
};

// ======================= Turing Traits ==============================
template <>
struct Int8GemmArchTraits<cutlass::arch::Sm75>
{
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
};

// ======================= Ampere Traits ==============================
template <>
struct Int8GemmArchTraits<cutlass::arch::Sm80>
{
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
