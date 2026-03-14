// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "MixedPrecisionCompat.h"

/*! \file SinSqDihedralForceGPU.cuh
    \brief Declares GPU kernel code for the sin²-multiplied dihedral force.
*/

#ifndef __SINSQDIHEDRALFORCEGPU_CUH__
#define __SINSQDIHEDRALFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver for the sin²-multiplied dihedral force
hipError_t gpu_compute_sinsq_dihedral_forces(ForceReal4* d_force,
                                              ForceReal* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const ForceReal4* d_pos,
                                              const BoxDim& box,
                                              const group_storage<4>* tlist,
                                              const unsigned int* dihedral_ABCD,
                                              const unsigned int pitch,
                                              const unsigned int* n_dihedrals_list,
                                              Scalar4* d_params,
                                              unsigned int n_dihedral_types,
                                              int block_size,
                                              int warp_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
