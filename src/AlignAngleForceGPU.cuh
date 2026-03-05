// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file AlignAngleForceGPU.cuh
    \brief Declares GPU kernel code for calculating the align-angle forces.
*/

#ifndef __ALIGNANGLEFORCEGPU_CUH__
#define __ALIGNANGLEFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes align angle forces for AlignAngleForceComputeGPU
hipError_t gpu_compute_align_angle_forces(Scalar4* d_force,
                                          Scalar4* d_torque,
                                          Scalar* d_virial,
                                          const size_t virial_pitch,
                                          const unsigned int N,
                                          const Scalar4* d_pos,
                                          const Scalar4* d_orientation,
                                          const BoxDim& box,
                                          const group_storage<3>* atable,
                                          const unsigned int* apos_list,
                                          const unsigned int pitch,
                                          const unsigned int* n_angles_list,
                                          const Scalar4* d_params,
                                          unsigned int n_angle_types,
                                          int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
