// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "AlignAngleForceGPU.cuh"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file AlignAngleForceGPU.cu
    \brief Defines GPU kernel code for computing the align-angle forces.

    Physics (angle group = (i, j, k)):
      d = minImage(r_k - r_j),  d_hat = d / |d|
      n_hat = rotate(q_i, (1,0,0))
      cos_theta = dot(n_hat, d_hat)
      U = (K/2) * (1 - cos_theta)
      tau_i = (K/2) * cross(n_hat, d_hat)
      F_j = -(K/2)/|d| * (n_hat - cos_theta*d_hat),  F_k = -F_j

    GPU parallelization: one thread per particle. Each thread loops over all
    angles it participates in and accumulates its own force/torque.
    'cur_angle_abc' tells the thread whether it is particle i(=0), j(=1), or k(=2).
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void gpu_compute_align_angle_forces_kernel(Scalar4* d_force,
                                                      Scalar4* d_torque,
                                                      Scalar* d_virial,
                                                      const size_t virial_pitch,
                                                      const unsigned int N,
                                                      const Scalar4* d_pos,
                                                      const Scalar4* d_orientation,
                                                      BoxDim box,
                                                      const group_storage<3>* alist,
                                                      const unsigned int* apos_list,
                                                      const unsigned int pitch,
                                                      const unsigned int* n_angles_list,
                                                      const Scalar* d_params)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= (int)N)
        return;

    int n_angles = n_angles_list[idx];

    // Position of this thread's particle
    Scalar4 idx_postype = d_pos[idx];
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);

    // Accumulated outputs for this thread's particle
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar4 torque_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
    Scalar virial[6];
    for (int v = 0; v < 6; v++)
        virial[v] = Scalar(0.0);

    // Body-frame reference axis (x-axis)
    vec3<Scalar> e_x(Scalar(1.0), Scalar(0.0), Scalar(0.0));

    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0]; // first "other" particle
        int cur_angle_y_idx = cur_angle.idx[1]; // second "other" particle
        int cur_angle_type = cur_angle.idx[2];

        // Which role does the current thread play? 0 = particle i, 1 = particle j, 2 = particle k
        int cur_angle_abc = apos_list[pitch * angle_idx + idx];

        // Load the two other particles' positions
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        // Determine positions of j, k based on which role this thread plays
        // The angle table stores: cur_angle.idx[0] and idx[1] are the OTHER two particles
        // apos_list tells which of i(0)/j(1)/k(2) this thread is
        Scalar3 pos_j, pos_k;
        int local_i_idx; // we need i's index to load its orientation

        if (cur_angle_abc == 0)
            {
            // This thread is particle i (the oriented one)
            pos_j = x_pos;
            pos_k = y_pos;
            local_i_idx = idx;
            }
        else if (cur_angle_abc == 1)
            {
            // This thread is particle j
            pos_j = idx_pos;
            pos_k = y_pos;
            local_i_idx = cur_angle_x_idx;
            }
        else // cur_angle_abc == 2
            {
            // This thread is particle k
            pos_k = idx_pos;
            pos_j = y_pos;
            local_i_idx = cur_angle_x_idx;
            }

        // Direction vector d = r_k - r_j (with minimum image)
        Scalar3 d;
        d.x = pos_k.x - pos_j.x;
        d.y = pos_k.y - pos_j.y;
        d.z = pos_k.z - pos_j.z;
        d = box.minImage(d);

        Scalar d_sq = d.x * d.x + d.y * d.y + d.z * d.z;
        Scalar d_mag = sqrtf(d_sq);

        if (d_mag < Scalar(1e-12))
            continue;

        Scalar d_inv = Scalar(1.0) / d_mag;
        vec3<Scalar> d_hat(d.x * d_inv, d.y * d_inv, d.z * d_inv);

        // Load orientation of particle i (the oriented particle)
        Scalar4 orientation_i = d_orientation[local_i_idx];
        quat<Scalar> q_i(orientation_i);

        // Body-frame x-axis in lab frame
        vec3<Scalar> n_hat = rotate(q_i, e_x);

        // cos(theta) = n_hat . d_hat
        Scalar cos_theta = dot(n_hat, d_hat);
        if (cos_theta > Scalar(1.0))
            cos_theta = Scalar(1.0);
        if (cos_theta < Scalar(-1.0))
            cos_theta = Scalar(-1.0);

        // Get parameter
        Scalar K = __ldg(d_params + cur_angle_type);

        // Energy: U = (K/2)(1 - cos_theta), split 1/3 per particle
        Scalar energy_third = K * (Scalar(1.0) - cos_theta) / Scalar(6.0);

        // Forces:
        // F_j = -(K/2)/|d| * (n_hat - cos_theta * d_hat)
        // F_k = -F_j
        vec3<Scalar> n_perp = n_hat - cos_theta * d_hat;
        vec3<Scalar> F_j = Scalar(-0.5) * K * d_inv * n_perp;

        // Torque on i: tau_i = (K/2) * cross(n_hat, d_hat)
        vec3<Scalar> tau_i = Scalar(0.5) * K * cross(n_hat, d_hat);

        // Virial: 1/3 of (F_j^a * d^b)
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * F_j.x * d.x;
        angle_virial[1] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.y * d.x + F_j.x * d.y);
        angle_virial[2] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.z * d.x + F_j.x * d.z);
        angle_virial[3] = Scalar(1. / 3.) * F_j.y * d.y;
        angle_virial[4] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.z * d.y + F_j.y * d.z);
        angle_virial[5] = Scalar(1. / 3.) * F_j.z * d.z;

        // Accumulate for THIS thread's particle
        if (cur_angle_abc == 0)
            {
            // This thread is particle i: gets torque only, no translational force
            torque_idx.x += tau_i.x;
            torque_idx.y += tau_i.y;
            torque_idx.z += tau_i.z;
            }
        else if (cur_angle_abc == 1)
            {
            // This thread is particle j: gets F_j
            force_idx.x += F_j.x;
            force_idx.y += F_j.y;
            force_idx.z += F_j.z;
            }
        else // cur_angle_abc == 2
            {
            // This thread is particle k: gets F_k = -F_j
            force_idx.x -= F_j.x;
            force_idx.y -= F_j.y;
            force_idx.z -= F_j.z;
            }

        force_idx.w += energy_third;

        for (int v = 0; v < 6; v++)
            virial[v] += angle_virial[v];
        }

    // Write out results
    d_force[idx] = force_idx;
    d_torque[idx] = torque_idx;
    for (int v = 0; v < 6; v++)
        d_virial[v * virial_pitch + idx] = virial[v];
    }

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
                                          const Scalar* d_params,
                                          unsigned int n_angle_types,
                                          int block_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_align_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min((unsigned int)block_size, max_block_size);

    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_align_angle_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_torque,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_orientation,
                       box,
                       atable,
                       apos_list,
                       pitch,
                       n_angles_list,
                       d_params);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
