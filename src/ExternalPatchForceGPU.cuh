// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file ExternalPatchForceGPU.cuh
    \brief Declares GPU kernel driver for computing external-patch forces.
*/

#ifndef __EXTERNALPATCHFORCEGPU_CUH__
#define __EXTERNALPATCHFORCEGPU_CUH__

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "MixedPrecisionCompat.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel driver that computes external-patch forces on the GPU
hipError_t gpu_compute_external_patch_forces(ForceReal4* d_force,
                                             ForceReal* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const unsigned int Nghosts,
                                             const ForceReal4* d_pos,
                                             const unsigned int* d_tag,
                                             const unsigned int* d_rtag,
                                             const int* d_partner_tags,
                                             const unsigned int partner_tags_size,
                                             const unsigned int* d_n_neigh,
                                             const unsigned int* d_nlist,
                                             const size_t* d_head_list,
                                             const BoxDim& box,
                                             const ForceReal epsilon,
                                             const ForceReal rcutsq,
                                             const ForceReal width,
                                             const int block_size,
                                             const int threads_per_particle);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __EXTERNALPATCHFORCEGPU_CUH__
