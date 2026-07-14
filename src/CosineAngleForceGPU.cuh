// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "MixedPrecisionCompat.h"

/*! \file CosineAngleForceGPU.cuh
    \brief Declares GPU kernel code for the cosine (worm-like-chain) angle force.
*/

#ifndef __COSINEANGLEFORCEGPU_CUH__
#define __COSINEANGLEFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the cosine (worm-like-chain) angle force
hipError_t gpu_compute_cosine_angle_forces(ForceReal4* d_force,
                                           ForceReal* d_virial,
                                           const size_t virial_pitch,
                                           const unsigned int N,
                                           const ForceReal4* d_pos,
                                           const BoxDim& box,
                                           const group_storage<3>* atable,
                                           const unsigned int* apos_list,
                                           const unsigned int pitch,
                                           const unsigned int* n_angles_list,
                                           Scalar4* d_params,
                                           unsigned int n_angle_types,
                                           int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
