// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/PotentialBondGPU.cuh"

#include "EvaluatorBondSoftHarmonic.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

template __attribute__((visibility("default"))) hipError_t
gpu_compute_bond_forces<EvaluatorBondSoftHarmonic, 2>(
    const kernel::bond_args_t<2>& bond_args,
    const typename EvaluatorBondSoftHarmonic::param_type* d_params,
    unsigned int* d_flags);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
