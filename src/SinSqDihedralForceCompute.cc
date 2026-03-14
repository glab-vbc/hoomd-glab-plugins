// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SinSqDihedralForceCompute.h"
#include "MixedPrecisionCompat.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file SinSqDihedralForceCompute.cc
    \brief Contains code for the SinSqDihedralForceCompute class

    Implements the sin²-multiplied dihedral potential:
      V = (k/2) * (1 + d * cos(n*phi - phi_0)) * sin²(theta_1) * sin²(theta_2)

    The forces are decomposed into three contributions:
      1. Dihedral torsion term (standard dihedral forces × S₁S₂, algebraically
         simplified to avoid 1/sin²θ singularity)
      2. Bond-angle θ₁ gradient term (affects atoms a, b, c)
      3. Bond-angle θ₂ gradient term (affects atoms b, c, d)

    All force expressions remain finite and smooth through collinear geometries.
*/

namespace hoomd
    {
namespace md
    {

SinSqDihedralForceCompute::SinSqDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_K(NULL), m_sign(NULL), m_multi(NULL), m_phi_0(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing SinSqDihedralForceCompute" << endl;

    m_dihedral_data = m_sysdef->getDihedralData();

    if (m_dihedral_data->getNTypes() == 0)
        {
        throw runtime_error("No dihedral types in the system.");
        }

    m_K = new Scalar[m_dihedral_data->getNTypes()];
    m_sign = new Scalar[m_dihedral_data->getNTypes()];
    m_multi = new int[m_dihedral_data->getNTypes()];
    m_phi_0 = new Scalar[m_dihedral_data->getNTypes()];
    }

SinSqDihedralForceCompute::~SinSqDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying SinSqDihedralForceCompute" << endl;

    delete[] m_K;
    delete[] m_sign;
    delete[] m_multi;
    delete[] m_phi_0;
    m_K = NULL;
    m_sign = NULL;
    m_multi = NULL;
    m_phi_0 = NULL;
    }

void SinSqDihedralForceCompute::setParams(unsigned int type,
                                           Scalar K,
                                           Scalar sign,
                                           int multiplicity,
                                           Scalar phi_0)
    {
    if (type >= m_dihedral_data->getNTypes())
        {
        throw runtime_error("Invalid dihedral type.");
        }

    m_K[type] = K;
    m_sign[type] = sign;
    m_multi[type] = multiplicity;
    m_phi_0[type] = phi_0;

    if (K <= 0)
        m_exec_conf->msg->warning() << "dihedral.sinsq: specified K <= 0" << endl;
    if (sign != 1 && sign != -1)
        m_exec_conf->msg->warning()
            << "dihedral.sinsq: a non-unitary sign was specified" << endl;
    }

void SinSqDihedralForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    sinsq_dihedral_params _params(params);
    setParams(typ, _params.k, _params.d, _params.n, _params.phi_0);
    }

pybind11::dict SinSqDihedralForceCompute::getParams(std::string type)
    {
    auto typ = m_dihedral_data->getTypeByName(type);
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["d"] = m_sign[typ];
    params["n"] = m_multi[typ];
    params["phi0"] = m_phi_0[typ];
    return params;
    }

void SinSqDihedralForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    // Access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<ForceReal> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // Zero output arrays
    m_force.zeroFill();
    m_virial.zeroFill();

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    size_t virial_pitch = m_virial.getPitch();
    const BoxDim& box = m_pdata->getBox();

    // Small epsilon for degenerate bond-length checks
    const Scalar SMALL = Scalar(1e-12);

    const unsigned int size = (unsigned int)m_dihedral_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // Lookup the tags of the four particles
        const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(i);
        assert(dihedral.tag[0] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[1] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[2] <= m_pdata->getMaximumTag());
        assert(dihedral.tag[3] <= m_pdata->getMaximumTag());

        unsigned int idx_a = h_rtag.data[dihedral.tag[0]];
        unsigned int idx_b = h_rtag.data[dihedral.tag[1]];
        unsigned int idx_c = h_rtag.data[dihedral.tag[2]];
        unsigned int idx_d = h_rtag.data[dihedral.tag[3]];

        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "dihedral.sinsq: dihedral " << dihedral.tag[0] << " " << dihedral.tag[1]
                << " " << dihedral.tag[2] << " " << dihedral.tag[3] << " incomplete." << endl
                << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // ===== Bond vectors with minimum image =====
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x;
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y;
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z;

        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);

        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;
        dcbm = box.minImage(dcbm);

        // ===== Cross products =====
        // A = dab × dcbm  (normal to plane a-b-c)
        Scalar aax = dab.y * dcbm.z - dab.z * dcbm.y;
        Scalar aay = dab.z * dcbm.x - dab.x * dcbm.z;
        Scalar aaz = dab.x * dcbm.y - dab.y * dcbm.x;

        // B = ddc × dcbm  (normal to plane b-c-d)
        Scalar bbx = ddc.y * dcbm.z - ddc.z * dcbm.y;
        Scalar bby = ddc.z * dcbm.x - ddc.x * dcbm.z;
        Scalar bbz = ddc.x * dcbm.y - ddc.y * dcbm.x;

        // Squared magnitudes
        Scalar raasq = aax * aax + aay * aay + aaz * aaz;
        Scalar rbbsq = bbx * bbx + bby * bby + bbz * bbz;
        Scalar rgsq = dcbm.x * dcbm.x + dcbm.y * dcbm.y + dcbm.z * dcbm.z;

        Scalar dab_sq = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar ddc_sq = ddc.x * ddc.x + ddc.y * ddc.y + ddc.z * ddc.z;

        // Skip degenerate geometries: zero bond lengths or exactly collinear
        // (sin²θ → 0 makes V and all forces exactly zero)
        if (raasq < SMALL || rbbsq < SMALL || dab_sq < SMALL || ddc_sq < SMALL || rgsq < SMALL)
            continue;

        Scalar rg = sqrt(rgsq);

        // ===== sin²θ factors =====
        // S₁ = sin²θ_abc = |A|² / (|dab|² |dcbm|²)
        // S₂ = sin²θ_bcd = |B|² / (|ddc|² |dcbm|²)
        Scalar S1 = raasq / (dab_sq * rgsq);
        Scalar S2 = rbbsq / (ddc_sq * rgsq);

        // ===== Dihedral angle via cross products =====
        Scalar rabinv = Scalar(1.0) / sqrt(raasq * rbbsq);
        Scalar c_abcd = (aax * bbx + aay * bby + aaz * bbz) * rabinv;
        Scalar s_abcd = rg * rabinv * (aax * ddc.x + aay * ddc.y + aaz * ddc.z);

        if (c_abcd > Scalar(1.0))
            c_abcd = Scalar(1.0);
        if (c_abcd < Scalar(-1.0))
            c_abcd = Scalar(-1.0);

        // ===== Parameters =====
        unsigned int dihedral_type = m_dihedral_data->getTypeByIndex(i);
        int multi = m_multi[dihedral_type];
        Scalar K = m_K[dihedral_type];
        Scalar sign = m_sign[dihedral_type];
        Scalar phi_0 = m_phi_0[dihedral_type];

        // ===== Chebyshev recurrence for cos(nφ), sin(nφ) =====
        Scalar p = Scalar(1.0);
        Scalar dfab = Scalar(0.0);
        Scalar ddfab = Scalar(0.0);

        for (int j = 0; j < multi; j++)
            {
            ddfab = p * c_abcd - dfab * s_abcd;
            dfab = p * s_abcd + dfab * c_abcd;
            p = ddfab;
            }

        // Apply phase shift
        Scalar sin_phi_0 = fast::sin(phi_0);
        Scalar cos_phi_0 = fast::cos(phi_0);
        p = p * cos_phi_0 + dfab * sin_phi_0;
        p = p * sign;
        dfab = dfab * cos_phi_0 - ddfab * sin_phi_0;
        dfab = dfab * sign;
        dfab *= (Scalar)-multi;
        p += Scalar(1.0);

        if (multi == 0)
            {
            p = Scalar(1.0) + sign;
            dfab = Scalar(0.0);
            }

        // V₀ = (K/2) * (1 + d·cos(nφ − φ₀)) = (K/2) * p
        Scalar V0 = K * Scalar(0.5) * p;

        // ===== Dot products for bond-angle terms =====
        Scalar fg = dab.x * dcbm.x + dab.y * dcbm.y + dab.z * dcbm.z;
        Scalar hg = ddc.x * dcbm.x + ddc.y * dcbm.y + ddc.z * dcbm.z;

        // ===================================================================
        // TERM 1: Dihedral torsion forces × S₁S₂
        //
        // The standard dihedral force on atom a is  df · gaa · A  where
        //   gaa = −r_g / |A|²
        // Multiplying by S₁S₂ cancels the singularity:
        //   gaa·S₁·S₂ = −|B|² · r_g / (|dab|² · |ddc|² · r_g⁴)
        //              = −|B|² / (|dab|² · |ddc|² · r_g³)
        // ===================================================================
        Scalar Rinv = Scalar(1.0) / (dab_sq * ddc_sq * rgsq * rgsq);
        Scalar rginv = Scalar(1.0) / rg;

        Scalar gaa_s = -rbbsq * rg * Rinv;             // gaa × S₁S₂
        Scalar gbb_s = raasq * rg * Rinv;              // gbb × S₁S₂
        Scalar fga_s = fg * rbbsq * Rinv * rginv;      // fga × S₁S₂
        Scalar hgb_s = hg * raasq * Rinv * rginv;      // hgb × S₁S₂

        Scalar dtfx = gaa_s * aax;
        Scalar dtfy = gaa_s * aay;
        Scalar dtfz = gaa_s * aaz;
        Scalar dtgx = fga_s * aax - hgb_s * bbx;
        Scalar dtgy = fga_s * aay - hgb_s * bby;
        Scalar dtgz = fga_s * aaz - hgb_s * bbz;
        Scalar dthx = gbb_s * bbx;
        Scalar dthy = gbb_s * bby;
        Scalar dthz = gbb_s * bbz;

        Scalar df = -K * dfab * Scalar(0.5);

        Scalar sx2 = df * dtgx;
        Scalar sy2 = df * dtgy;
        Scalar sz2 = df * dtgz;

        // Dihedral-term force on atom a
        Scalar f1ax = df * dtfx;
        Scalar f1ay = df * dtfy;
        Scalar f1az = df * dtfz;

        // Dihedral-term force on atom b
        Scalar f1bx = sx2 - f1ax;
        Scalar f1by = sy2 - f1ay;
        Scalar f1bz = sz2 - f1az;

        // Dihedral-term force on atom d
        Scalar f1dx = df * dthx;
        Scalar f1dy = df * dthy;
        Scalar f1dz = df * dthz;

        // Dihedral-term force on atom c
        Scalar f1cx = -sx2 - f1dx;
        Scalar f1cy = -sy2 - f1dy;
        Scalar f1cz = -sz2 - f1dz;

        // ===================================================================
        // TERM 2: Gradient of S₁ = sin²θ_abc (affects atoms a, b, c only)
        //
        // S₁ = 1 − fg² / (dab_sq · rgsq)   where fg = dab · dcbm
        //
        // ∂S₁/∂r_a = −(2fg / (dab_sq · rgsq)) · (dcbm − (fg/dab_sq)·dab)
        // ∂S₁/∂r_c =  (2fg / (dab_sq · rgsq)) · (dab  − (fg/rgsq)·dcbm)
        // ∂S₁/∂r_b = −(∂S₁/∂r_a + ∂S₁/∂r_c)
        //
        // Force contribution: F_i^(2) = −V₀ · S₂ · ∂S₁/∂r_i
        // ===================================================================
        Scalar pref_s1 = Scalar(2.0) * fg / (dab_sq * rgsq);
        Scalar V0_S2 = V0 * S2;

        Scalar fg_over_dabsq = fg / dab_sq;
        Scalar ds1a_x = dcbm.x - fg_over_dabsq * dab.x;
        Scalar ds1a_y = dcbm.y - fg_over_dabsq * dab.y;
        Scalar ds1a_z = dcbm.z - fg_over_dabsq * dab.z;

        Scalar fg_over_rgsq = fg / rgsq;
        Scalar ds1c_x = dab.x - fg_over_rgsq * dcbm.x;
        Scalar ds1c_y = dab.y - fg_over_rgsq * dcbm.y;
        Scalar ds1c_z = dab.z - fg_over_rgsq * dcbm.z;

        // F_a^(2) = −V₀·S₂·(∂S₁/∂r_a) = V₀·S₂·pref_s1·ds1a
        Scalar f2ax = V0_S2 * pref_s1 * ds1a_x;
        Scalar f2ay = V0_S2 * pref_s1 * ds1a_y;
        Scalar f2az = V0_S2 * pref_s1 * ds1a_z;

        // F_c^(2) = −V₀·S₂·(∂S₁/∂r_c) = −V₀·S₂·pref_s1·ds1c
        Scalar f2cx = -V0_S2 * pref_s1 * ds1c_x;
        Scalar f2cy = -V0_S2 * pref_s1 * ds1c_y;
        Scalar f2cz = -V0_S2 * pref_s1 * ds1c_z;

        // F_b^(2) = −(F_a^(2) + F_c^(2))   (translational invariance)
        Scalar f2bx = -(f2ax + f2cx);
        Scalar f2by = -(f2ay + f2cy);
        Scalar f2bz = -(f2az + f2cz);

        // ===================================================================
        // TERM 3: Gradient of S₂ = sin²θ_bcd (affects atoms b, c, d only)
        //
        // S₂ = 1 − hg² / (ddc_sq · rgsq)   where hg = ddc · dcbm
        //
        // ∂S₂/∂r_d = −(2hg / (ddc_sq · rgsq)) · (dcbm − (hg/ddc_sq)·ddc)
        // ∂S₂/∂r_b = −(2hg / (ddc_sq · rgsq)) · (ddc  − (hg/rgsq)·dcbm)
        //   Note: both have the same sign because r_b enters dcbm = r_b − r_c
        //   with a + sign (unlike r_c which enters with a − sign in S₁).
        // ∂S₂/∂r_c = −(∂S₂/∂r_d + ∂S₂/∂r_b)
        //
        // Force contribution: F_i^(3) = −V₀ · S₁ · ∂S₂/∂r_i
        // ===================================================================
        Scalar pref_s2 = Scalar(2.0) * hg / (ddc_sq * rgsq);
        Scalar V0_S1 = V0 * S1;

        Scalar hg_over_ddcsq = hg / ddc_sq;
        Scalar ds2d_x = dcbm.x - hg_over_ddcsq * ddc.x;
        Scalar ds2d_y = dcbm.y - hg_over_ddcsq * ddc.y;
        Scalar ds2d_z = dcbm.z - hg_over_ddcsq * ddc.z;

        Scalar hg_over_rgsq = hg / rgsq;
        Scalar ds2b_x = ddc.x - hg_over_rgsq * dcbm.x;
        Scalar ds2b_y = ddc.y - hg_over_rgsq * dcbm.y;
        Scalar ds2b_z = ddc.z - hg_over_rgsq * dcbm.z;

        // F_d^(3) = −V₀·S₁·(∂S₂/∂r_d) = V₀·S₁·pref_s2·ds2d
        Scalar f3dx = V0_S1 * pref_s2 * ds2d_x;
        Scalar f3dy = V0_S1 * pref_s2 * ds2d_y;
        Scalar f3dz = V0_S1 * pref_s2 * ds2d_z;

        // ∂S₂/∂r_b = −pref_s2·ds2b  (note: sign differs from ∂S₁/∂r_c
        //   because r_b enters dcbm with a + sign)
        // F_b^(3) = −V₀·S₁·(∂S₂/∂r_b) = +V₀·S₁·pref_s2·ds2b
        Scalar f3bx = V0_S1 * pref_s2 * ds2b_x;
        Scalar f3by = V0_S1 * pref_s2 * ds2b_y;
        Scalar f3bz = V0_S1 * pref_s2 * ds2b_z;

        // F_c^(3) = −(F_d^(3) + F_b^(3))   (translational invariance)
        Scalar f3cx = -(f3dx + f3bx);
        Scalar f3cy = -(f3dy + f3by);
        Scalar f3cz = -(f3dz + f3bz);

        // ===================================================================
        // Total forces: sum of all three terms
        // ===================================================================
        Scalar Fax = f1ax + f2ax;
        Scalar Fay = f1ay + f2ay;
        Scalar Faz = f1az + f2az;

        Scalar Fbx = f1bx + f2bx + f3bx;
        Scalar Fby = f1by + f2by + f3by;
        Scalar Fbz = f1bz + f2bz + f3bz;

        Scalar Fcx = f1cx + f2cx + f3cx;
        Scalar Fcy = f1cy + f2cy + f3cy;
        Scalar Fcz = f1cz + f2cz + f3cz;

        Scalar Fdx = f1dx + f3dx;
        Scalar Fdy = f1dy + f3dy;
        Scalar Fdz = f1dz + f3dz;

        // Energy: 1/4 per atom
        Scalar dihedral_eng = V0 * S1 * S2 * Scalar(0.25);

        // Virial: 1/4 per atom, upper-triangular, using total forces
        Scalar ddcb_x = ddc.x + dcb.x;
        Scalar ddcb_y = ddc.y + dcb.y;
        Scalar ddcb_z = ddc.z + dcb.z;

        Scalar dihedral_virial[6];
        dihedral_virial[0] = Scalar(0.25) * (dab.x * Fax + dcb.x * Fcx + ddcb_x * Fdx);
        dihedral_virial[1] = Scalar(0.25) * (dab.y * Fax + dcb.y * Fcx + ddcb_y * Fdx);
        dihedral_virial[2] = Scalar(0.25) * (dab.z * Fax + dcb.z * Fcx + ddcb_z * Fdx);
        dihedral_virial[3] = Scalar(0.25) * (dab.y * Fay + dcb.y * Fcy + ddcb_y * Fdy);
        dihedral_virial[4] = Scalar(0.25) * (dab.z * Fay + dcb.z * Fcy + ddcb_z * Fdy);
        dihedral_virial[5] = Scalar(0.25) * (dab.z * Faz + dcb.z * Fcz + ddcb_z * Fdz);

        // Accumulate on each atom
        h_force.data[idx_a].x += Fax;
        h_force.data[idx_a].y += Fay;
        h_force.data[idx_a].z += Faz;
        h_force.data[idx_a].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_a] += dihedral_virial[k];

        h_force.data[idx_b].x += Fbx;
        h_force.data[idx_b].y += Fby;
        h_force.data[idx_b].z += Fbz;
        h_force.data[idx_b].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_b] += dihedral_virial[k];

        h_force.data[idx_c].x += Fcx;
        h_force.data[idx_c].y += Fcy;
        h_force.data[idx_c].z += Fcz;
        h_force.data[idx_c].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_c] += dihedral_virial[k];

        h_force.data[idx_d].x += Fdx;
        h_force.data[idx_d].y += Fdy;
        h_force.data[idx_d].z += Fdz;
        h_force.data[idx_d].w += dihedral_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[virial_pitch * k + idx_d] += dihedral_virial[k];
        }
    }

namespace detail
    {
void export_SinSqDihedralForceCompute(pybind11::module& m)
    {
    pybind11::class_<SinSqDihedralForceCompute,
                     ForceCompute,
                     std::shared_ptr<SinSqDihedralForceCompute>>(m,
                                                                  "SinSqDihedralForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &SinSqDihedralForceCompute::setParamsPython)
        .def("getParams", &SinSqDihedralForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
