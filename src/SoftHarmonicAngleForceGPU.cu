// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SoftHarmonicAngleForceGPU.cuh"
#include "SoftHarmonicTail.h"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL ForceReal(0.001)

/*! \file SoftHarmonicAngleForceGPU.cu
    \brief GPU kernel for the soft/capped harmonic angle force.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel for calculating soft/capped harmonic angle forces on the GPU
__global__ void
gpu_compute_soft_harmonic_angle_forces_kernel(ForceReal4* d_force,
                                              ForceReal* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int N,
                                              const ForceReal4* d_pos,
                                              const Scalar4* d_params,
                                              BoxDim box,
                                              const group_storage<3>* alist,
                                              const unsigned int* apos_list,
                                              const unsigned int pitch,
                                              const unsigned int* n_angles_list)
    {
    // identify the particle this thread handles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int n_angles = n_angles_list[idx];

    // this particle can be a, b, or c in any of its a-b-c triplets
    ForceReal4 idx_postype = d_pos[idx];
    ForceReal3 idx_pos
        = make_forcereal3(ForceReal(idx_postype.x), ForceReal(idx_postype.y), ForceReal(idx_postype.z));
    ForceReal3 a_pos, b_pos, c_pos;

    ForceReal4 force_idx = make_forcereal4(ForceReal(0.0), ForceReal(0.0), ForceReal(0.0), ForceReal(0.0));
    ForceReal fab[3], fcb[3];

    ForceReal virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = ForceReal(0.0);

    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];
        int cur_angle_type = cur_angle.idx[2];

        int cur_angle_abc = apos_list[pitch * angle_idx + idx];

        ForceReal4 x_postype = d_pos[cur_angle_x_idx];
        ForceReal3 x_pos = make_forcereal3(ForceReal(x_postype.x), ForceReal(x_postype.y), ForceReal(x_postype.z));
        ForceReal4 y_postype = d_pos[cur_angle_y_idx];
        ForceReal3 y_pos = make_forcereal3(ForceReal(y_postype.x), ForceReal(y_postype.y), ForceReal(y_postype.z));

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        ForceReal3 dab = a_pos - b_pos;
        ForceReal3 dcb = c_pos - b_pos;

#ifdef HOOMD_HAS_FORCEREAL
        dab = box.minImageForceReal(dab);
        dcb = box.minImageForceReal(dcb);
#else
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
#endif

        // load the angle parameters
        Scalar4 params = __ldg(d_params + cur_angle_type);
        ForceReal K = ForceReal(params.x);
        ForceReal t_0 = ForceReal(params.y);
        ForceReal x_c = ForceReal(params.z);
        int mode = (params.w > Scalar(0.5)) ? SOFT_HARMONIC_FLAT : SOFT_HARMONIC_LINEAR;

        ForceReal rsqab = dot(dab, dab);
        ForceReal rab = sqrtf(rsqab);
        ForceReal rsqcb = dot(dcb, dcb);
        ForceReal rcb = sqrtf(rsqcb);

        ForceReal c_abbc = dot(dab, dcb);
        c_abbc /= rab * rcb;

        if (c_abbc > ForceReal(1.0))
            c_abbc = ForceReal(1.0);
        if (c_abbc < -ForceReal(1.0))
            c_abbc = -ForceReal(1.0);

        ForceReal s_abbc = sqrtf(ForceReal(1.0) - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = ForceReal(1.0) / s_abbc;

        // evaluate the potential
        ForceReal dth = fast::acos(c_abbc) - t_0;
        ForceReal dUdth, U;
        softHarmonicTail<ForceReal>(mode, K, x_c, dth, dUdth, U);

        ForceReal a = -ForceReal(1.0) * dUdth * s_abbc;
        ForceReal a11 = a * c_abbc / rsqab;
        ForceReal a12 = -a / (rab * rcb);
        ForceReal a22 = a * c_abbc / rsqcb;

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        // 1/3 of the energy to each of the three atoms in the angle
        ForceReal angle_eng = U * ForceReal(ForceReal(1.0) / ForceReal(3.0));

        // upper triangular version of virial tensor, 1/3 to each atom
        ForceReal angle_virial[6];
        angle_virial[0] = ForceReal(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = ForceReal(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = ForceReal(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = ForceReal(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = ForceReal(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = ForceReal(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    // write out the result
    d_force[idx] = force_idx;
    for (int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = ForceReal(virial[i]);
    }

hipError_t gpu_compute_soft_harmonic_angle_forces(ForceReal4* d_force,
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
                                                  int block_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_soft_harmonic_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_soft_harmonic_angle_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       atable,
                       apos_list,
                       pitch,
                       n_angles_list);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
