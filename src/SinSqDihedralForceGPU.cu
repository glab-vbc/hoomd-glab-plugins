// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SinSqDihedralForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

#if HOOMD_LONGREAL_SIZE == 32
#define __scalar2int_rn __float2int_rn
#else
#define __scalar2int_rn __double2int_rn
#endif

// Small epsilon for degenerate bond-length checks
#define SMALL_SINSQ ForceReal(1e-12)

/*! \file SinSqDihedralForceGPU.cu
    \brief GPU kernel for the sin²-multiplied dihedral force.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void gpu_compute_sinsq_dihedral_forces_kernel(ForceReal4* d_force,
                                                          ForceReal* d_virial,
                                                          const size_t virial_pitch,
                                                          const unsigned int N,
                                                          const ForceReal4* d_pos,
                                                          const Scalar4* d_params,
                                                          BoxDim box,
                                                          const group_storage<4>* tlist,
                                                          const unsigned int* dihedral_ABCD,
                                                          const unsigned int pitch,
                                                          const unsigned int* n_dihedrals_list)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    int n_dihedrals = n_dihedrals_list[idx];

    ForceReal4 idx_postype = d_pos[idx];
    ForceReal3 idx_pos = make_forcereal3(ForceReal(idx_postype.x),
                                         ForceReal(idx_postype.y),
                                         ForceReal(idx_postype.z));

    // Initialize per-thread accumulators
    ForceReal4 force_idx = make_forcereal4(ForceReal(0.0), ForceReal(0.0),
                                           ForceReal(0.0), ForceReal(0.0));
    ForceReal virial_idx[6];
    for (unsigned int vi = 0; vi < 6; vi++)
        virial_idx[vi] = ForceReal(0.0);

    for (int dihedral_idx = 0; dihedral_idx < n_dihedrals; dihedral_idx++)
        {
        group_storage<4> cur_dihedral = tlist[pitch * dihedral_idx + idx];
        unsigned int cur_ABCD = dihedral_ABCD[pitch * dihedral_idx + idx];

        int cur_dihedral_x_idx = cur_dihedral.idx[0];
        int cur_dihedral_y_idx = cur_dihedral.idx[1];
        int cur_dihedral_z_idx = cur_dihedral.idx[2];
        int cur_dihedral_type = cur_dihedral.idx[3];
        int cur_dihedral_abcd = cur_ABCD;

        ForceReal4 x_postype = d_pos[cur_dihedral_x_idx];
        ForceReal3 x_pos = make_forcereal3(ForceReal(x_postype.x), ForceReal(x_postype.y), ForceReal(x_postype.z));
        ForceReal4 y_postype = d_pos[cur_dihedral_y_idx];
        ForceReal3 y_pos = make_forcereal3(ForceReal(y_postype.x), ForceReal(y_postype.y), ForceReal(y_postype.z));
        ForceReal4 z_postype = d_pos[cur_dihedral_z_idx];
        ForceReal3 z_pos = make_forcereal3(ForceReal(z_postype.x), ForceReal(z_postype.y), ForceReal(z_postype.z));

        ForceReal3 pos_a, pos_b, pos_c, pos_d;

        if (cur_dihedral_abcd == 0)
            { pos_a = idx_pos; pos_b = x_pos; pos_c = y_pos; pos_d = z_pos; }
        if (cur_dihedral_abcd == 1)
            { pos_b = idx_pos; pos_a = x_pos; pos_c = y_pos; pos_d = z_pos; }
        if (cur_dihedral_abcd == 2)
            { pos_c = idx_pos; pos_a = x_pos; pos_b = y_pos; pos_d = z_pos; }
        if (cur_dihedral_abcd == 3)
            { pos_d = idx_pos; pos_a = x_pos; pos_b = y_pos; pos_c = z_pos; }

        // Bond vectors
        ForceReal3 dab = pos_a - pos_b;
        ForceReal3 dcb = pos_c - pos_b;
        ForceReal3 ddc = pos_d - pos_c;

#ifdef HOOMD_HAS_FORCEREAL
        dab = box.minImageForceReal(dab);
        dcb = box.minImageForceReal(dcb);
        ddc = box.minImageForceReal(ddc);
#else
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);
#endif

        ForceReal3 dcbm = -dcb;
#ifdef HOOMD_HAS_FORCEREAL
        dcbm = box.minImageForceReal(dcbm);
#else
        dcbm = box.minImage(dcbm);
#endif

        // Load parameters
        Scalar4 params = __ldg(d_params + cur_dihedral_type);
        ForceReal K = ForceReal(params.x);
        ForceReal sign = ForceReal(params.y);
        ForceReal multi = ForceReal(params.z);
        ForceReal phi_0 = ForceReal(params.w);

        // Cross products
        ForceReal aax = dab.y * dcbm.z - dab.z * dcbm.y;
        ForceReal aay = dab.z * dcbm.x - dab.x * dcbm.z;
        ForceReal aaz = dab.x * dcbm.y - dab.y * dcbm.x;

        ForceReal bbx = ddc.y * dcbm.z - ddc.z * dcbm.y;
        ForceReal bby = ddc.z * dcbm.x - ddc.x * dcbm.z;
        ForceReal bbz = ddc.x * dcbm.y - ddc.y * dcbm.x;

        ForceReal raasq = aax * aax + aay * aay + aaz * aaz;
        ForceReal rbbsq = bbx * bbx + bby * bby + bbz * bbz;
        ForceReal rgsq = dcbm.x * dcbm.x + dcbm.y * dcbm.y + dcbm.z * dcbm.z;

        ForceReal dab_sq = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        ForceReal ddc_sq = ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z;

        // Skip degenerate geometries
        if (raasq < SMALL_SINSQ || rbbsq < SMALL_SINSQ ||
            dab_sq < SMALL_SINSQ || ddc_sq < SMALL_SINSQ || rgsq < SMALL_SINSQ)
            continue;

        ForceReal rg = fast::sqrt(rgsq);

        // sin²θ factors
        ForceReal S1 = raasq / (dab_sq * rgsq);
        ForceReal S2 = rbbsq / (ddc_sq * rgsq);

        // Dihedral angle
        ForceReal rabinv = ForceReal(1.0) / fast::sqrt(raasq * rbbsq);
        ForceReal c_abcd = (aax * bbx + aay * bby + aaz * bbz) * rabinv;
        ForceReal s_abcd = rg * rabinv * (aax * ddc.x + aay * ddc.y + aaz * ddc.z);

        if (c_abcd > ForceReal(1.0)) c_abcd = ForceReal(1.0);
        if (c_abcd < -ForceReal(1.0)) c_abcd = -ForceReal(1.0);
        if (s_abcd > ForceReal(1.0)) s_abcd = ForceReal(1.0);
        if (s_abcd < -ForceReal(1.0)) s_abcd = -ForceReal(1.0);

        // Chebyshev recurrence
        ForceReal p = ForceReal(1.0);
        ForceReal ddfab;
        ForceReal dfab = ForceReal(0.0);
        int m = __scalar2int_rn(params.z);

        for (int jj = 0; jj < m; jj++)
            {
            ddfab = p * c_abcd - dfab * s_abcd;
            dfab = p * s_abcd + dfab * c_abcd;
            p = ddfab;
            }

        // Phase shift
        ForceReal sin_phi_0 = fast::sin(phi_0);
        ForceReal cos_phi_0 = fast::cos(phi_0);
        p = p * cos_phi_0 + dfab * sin_phi_0;
        p *= sign;
        dfab = dfab * cos_phi_0 - ddfab * sin_phi_0;
        dfab *= sign;
        dfab *= -multi;
        p += ForceReal(1.0);

        if (multi < ForceReal(1.0))
            {
            p = ForceReal(1.0) + sign;
            dfab = ForceReal(0.0);
            }

        ForceReal V0 = K * ForceReal(0.5) * p;

        // Dot products for angle terms
        ForceReal fg = dab.x * dcbm.x + dab.y * dcbm.y + dab.z * dcbm.z;
        ForceReal hg = ddc.x * dcbm.x + ddc.y * dcbm.y + ddc.z * dcbm.z;

        // ===== TERM 1: Dihedral torsion × S₁S₂ =====
        ForceReal Rinv = ForceReal(1.0) / (dab_sq * ddc_sq * rgsq * rgsq);
        ForceReal rginv = ForceReal(1.0) / rg;

        ForceReal gaa_s = -rbbsq * rg * Rinv;
        ForceReal gbb_s = raasq * rg * Rinv;
        ForceReal fga_s = fg * rbbsq * Rinv * rginv;
        ForceReal hgb_s = hg * raasq * Rinv * rginv;

        ForceReal dtfx = gaa_s * aax;
        ForceReal dtfy = gaa_s * aay;
        ForceReal dtfz = gaa_s * aaz;
        ForceReal dtgx = fga_s * aax - hgb_s * bbx;
        ForceReal dtgy = fga_s * aay - hgb_s * bby;
        ForceReal dtgz = fga_s * aaz - hgb_s * bbz;
        ForceReal dthx = gbb_s * bbx;
        ForceReal dthy = gbb_s * bby;
        ForceReal dthz = gbb_s * bbz;

        ForceReal df = -K * dfab * ForceReal(0.5);

        ForceReal sx2 = df * dtgx;
        ForceReal sy2 = df * dtgy;
        ForceReal sz2 = df * dtgz;

        ForceReal f1ax = df * dtfx;
        ForceReal f1ay = df * dtfy;
        ForceReal f1az = df * dtfz;

        ForceReal f1bx = sx2 - f1ax;
        ForceReal f1by = sy2 - f1ay;
        ForceReal f1bz = sz2 - f1az;

        ForceReal f1dx = df * dthx;
        ForceReal f1dy = df * dthy;
        ForceReal f1dz = df * dthz;

        ForceReal f1cx = -sx2 - f1dx;
        ForceReal f1cy = -sy2 - f1dy;
        ForceReal f1cz = -sz2 - f1dz;

        // ===== TERM 2: ∂S₁/∂r =====
        ForceReal pref_s1 = ForceReal(2.0) * fg / (dab_sq * rgsq);
        ForceReal V0_S2 = V0 * S2;

        ForceReal fg_over_dabsq = fg / dab_sq;
        ForceReal ds1a_x = dcbm.x - fg_over_dabsq * dab.x;
        ForceReal ds1a_y = dcbm.y - fg_over_dabsq * dab.y;
        ForceReal ds1a_z = dcbm.z - fg_over_dabsq * dab.z;

        ForceReal fg_over_rgsq = fg / rgsq;
        ForceReal ds1c_x = dab.x - fg_over_rgsq * dcbm.x;
        ForceReal ds1c_y = dab.y - fg_over_rgsq * dcbm.y;
        ForceReal ds1c_z = dab.z - fg_over_rgsq * dcbm.z;

        ForceReal f2ax = V0_S2 * pref_s1 * ds1a_x;
        ForceReal f2ay = V0_S2 * pref_s1 * ds1a_y;
        ForceReal f2az = V0_S2 * pref_s1 * ds1a_z;

        ForceReal f2cx = -V0_S2 * pref_s1 * ds1c_x;
        ForceReal f2cy = -V0_S2 * pref_s1 * ds1c_y;
        ForceReal f2cz = -V0_S2 * pref_s1 * ds1c_z;

        ForceReal f2bx = -(f2ax + f2cx);
        ForceReal f2by = -(f2ay + f2cy);
        ForceReal f2bz = -(f2az + f2cz);

        // ===== TERM 3: ∂S₂/∂r =====
        ForceReal pref_s2 = ForceReal(2.0) * hg / (ddc_sq * rgsq);
        ForceReal V0_S1 = V0 * S1;

        ForceReal hg_over_ddcsq = hg / ddc_sq;
        ForceReal ds2d_x = dcbm.x - hg_over_ddcsq * ddc.x;
        ForceReal ds2d_y = dcbm.y - hg_over_ddcsq * ddc.y;
        ForceReal ds2d_z = dcbm.z - hg_over_ddcsq * ddc.z;

        ForceReal hg_over_rgsq = hg / rgsq;
        ForceReal ds2b_x = ddc.x - hg_over_rgsq * dcbm.x;
        ForceReal ds2b_y = ddc.y - hg_over_rgsq * dcbm.y;
        ForceReal ds2b_z = ddc.z - hg_over_rgsq * dcbm.z;

        ForceReal f3dx = V0_S1 * pref_s2 * ds2d_x;
        ForceReal f3dy = V0_S1 * pref_s2 * ds2d_y;
        ForceReal f3dz = V0_S1 * pref_s2 * ds2d_z;

        // ∂S₂/∂r_b = −pref_s2·ds2b  (sign differs from ∂S₁/∂r_c)
        ForceReal f3bx = V0_S1 * pref_s2 * ds2b_x;
        ForceReal f3by = V0_S1 * pref_s2 * ds2b_y;
        ForceReal f3bz = V0_S1 * pref_s2 * ds2b_z;

        ForceReal f3cx = -(f3dx + f3bx);
        ForceReal f3cy = -(f3dy + f3by);
        ForceReal f3cz = -(f3dz + f3bz);

        // ===== Total forces =====
        ForceReal Fax = f1ax + f2ax;
        ForceReal Fay = f1ay + f2ay;
        ForceReal Faz = f1az + f2az;

        ForceReal Fbx = f1bx + f2bx + f3bx;
        ForceReal Fby = f1by + f2by + f3by;
        ForceReal Fbz = f1bz + f2bz + f3bz;

        ForceReal Fcx = f1cx + f2cx + f3cx;
        ForceReal Fcy = f1cy + f2cy + f3cy;
        ForceReal Fcz = f1cz + f2cz + f3cz;

        ForceReal Fdx = f1dx + f3dx;
        ForceReal Fdy = f1dy + f3dy;
        ForceReal Fdz = f1dz + f3dz;

        // Energy: 1/4 per atom
        ForceReal dihedral_eng = V0 * S1 * S2 * ForceReal(0.25);

        // Virial: 1/4 per atom
        ForceReal ddcb_x = ddc.x + dcb.x;
        ForceReal ddcb_y = ddc.y + dcb.y;
        ForceReal ddcb_z = ddc.z + dcb.z;

        ForceReal dihedral_virial[6];
        dihedral_virial[0] = ForceReal(0.25) * (dab.x * Fax + dcb.x * Fcx + ddcb_x * Fdx);
        dihedral_virial[1] = ForceReal(0.25) * (dab.y * Fax + dcb.y * Fcx + ddcb_y * Fdx);
        dihedral_virial[2] = ForceReal(0.25) * (dab.z * Fax + dcb.z * Fcx + ddcb_z * Fdx);
        dihedral_virial[3] = ForceReal(0.25) * (dab.y * Fay + dcb.y * Fcy + ddcb_y * Fdy);
        dihedral_virial[4] = ForceReal(0.25) * (dab.z * Fay + dcb.z * Fcy + ddcb_z * Fdy);
        dihedral_virial[5] = ForceReal(0.25) * (dab.z * Faz + dcb.z * Fcz + ddcb_z * Fdz);

        // Accumulate this thread's contribution
        if (cur_dihedral_abcd == 0)
            { force_idx.x += Fax; force_idx.y += Fay; force_idx.z += Faz; }
        if (cur_dihedral_abcd == 1)
            { force_idx.x += Fbx; force_idx.y += Fby; force_idx.z += Fbz; }
        if (cur_dihedral_abcd == 2)
            { force_idx.x += Fcx; force_idx.y += Fcy; force_idx.z += Fcz; }
        if (cur_dihedral_abcd == 3)
            { force_idx.x += Fdx; force_idx.y += Fdy; force_idx.z += Fdz; }

        force_idx.w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            virial_idx[k] += dihedral_virial[k];
        }

    // Write accumulated result
    d_force[idx] = force_idx;
    for (int k = 0; k < 6; k++)
        d_virial[k * virial_pitch + idx] = virial_idx[k];
    }

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
                                              int warp_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_sinsq_dihedral_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    if (max_block_size % warp_size)
        max_block_size = (max_block_size / warp_size - 1) * warp_size;

    unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_sinsq_dihedral_forces_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       tlist,
                       dihedral_ABCD,
                       pitch,
                       n_dihedrals_list);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
