// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "AlignAngleForceCompute.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file AlignAngleForceCompute.cc
    \brief Contains code for the AlignAngleForceCompute class.

    Physics:
    For angle group (i, j, k):
      d = minImage(r_k - r_j),  d_hat = d / |d|
      n_hat = rotate(q_i, (1,0,0))   [body-frame x-axis of particle i]
      cos_theta = dot(n_hat, d_hat)

      U = (k/2) * (1 - cos_theta)

    Torque on i (in lab frame):
      tau_i = (k/2) * cross(n_hat, d_hat)

    Forces on j and k (from dependence of d_hat on positions):
      Let P = (I - d_hat ⊗ d_hat) / |d|   [projector ⊥ d_hat, scaled by 1/|d|]
      F_j = -(k/2) * P . n_hat = -(k/2)/|d| * (n_hat - cos_theta * d_hat)
      F_k = -F_j
      (No force on i from this potential — it only couples to i's orientation.)
*/

namespace hoomd
    {
namespace md
    {

AlignAngleForceCompute::AlignAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_K(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing AlignAngleForceCompute" << endl;

    m_angle_data = m_sysdef->getAngleData();

    if (m_angle_data->getNTypes() == 0)
        {
        throw runtime_error("No angle types in the system.");
        }

    m_K = new Scalar[m_angle_data->getNTypes()];
    memset(m_K, 0, sizeof(Scalar) * m_angle_data->getNTypes());
    }

AlignAngleForceCompute::~AlignAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying AlignAngleForceCompute" << endl;
    delete[] m_K;
    m_K = NULL;
    }

void AlignAngleForceCompute::setParams(unsigned int type, Scalar K)
    {
    if (type >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }
    m_K[type] = K;

    if (K <= 0)
        m_exec_conf->msg->warning() << "angle.align: specified K <= 0" << endl;
    }

void AlignAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_angle_data->getTypeByName(type);
    auto _params = align_angle_params(params);
    setParams(typ, _params.k);
    }

pybind11::dict AlignAngleForceCompute::getParams(std::string type)
    {
    auto typ = m_angle_data->getTypeByName(type);
    if (typ >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    return params;
    }

void AlignAngleForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    // Access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // Access force/torque/virial output
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    // Zero output arrays
    m_force.zeroFill();
    m_torque.zeroFill();
    m_virial.zeroFill();

    // Box for periodic boundary conditions
    const BoxDim& box = m_pdata->getGlobalBox();

    // Body-frame reference axis (x-axis)
    const vec3<Scalar> e_x(1.0, 0.0, 0.0);

    const unsigned int size = (unsigned int)m_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // Get the three member tags
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[2] <= m_pdata->getMaximumTag());

        // Convert tags to local indices
        unsigned int idx_i = h_rtag.data[angle.tag[0]];
        unsigned int idx_j = h_rtag.data[angle.tag[1]];
        unsigned int idx_k = h_rtag.data[angle.tag[2]];

        // Check completeness
        if (idx_i == NOT_LOCAL || idx_j == NOT_LOCAL || idx_k == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "angle.align: angle " << angle.tag[0] << " " << angle.tag[1] << " "
                << angle.tag[2] << " incomplete." << endl;
            throw std::runtime_error("Error in align angle calculation");
            }

        assert(idx_i < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_j < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_k < m_pdata->getN() + m_pdata->getNGhosts());

        // Get positions
        Scalar3 pos_j = make_scalar3(h_pos.data[idx_j].x, h_pos.data[idx_j].y, h_pos.data[idx_j].z);
        Scalar3 pos_k = make_scalar3(h_pos.data[idx_k].x, h_pos.data[idx_k].y, h_pos.data[idx_k].z);

        // Direction vector d = r_k - r_j  (with minimum image)
        Scalar3 d;
        d.x = pos_k.x - pos_j.x;
        d.y = pos_k.y - pos_j.y;
        d.z = pos_k.z - pos_j.z;
        d = box.minImage(d);

        Scalar d_sq = d.x * d.x + d.y * d.y + d.z * d.z;
        Scalar d_mag = sqrt(d_sq);

        // Avoid division by zero for degenerate cases
        if (d_mag < Scalar(1e-12))
            continue;

        Scalar d_inv = Scalar(1.0) / d_mag;
        vec3<Scalar> d_hat(d.x * d_inv, d.y * d_inv, d.z * d_inv);

        // Get orientation of particle i (the oriented particle)
        quat<Scalar> q_i(h_orientation.data[idx_i]);

        // Compute body-frame x-axis in lab frame
        vec3<Scalar> n_hat = rotate(q_i, e_x);

        // cos(theta) = n_hat . d_hat
        Scalar cos_theta = dot(n_hat, d_hat);

        // Clamp for safety
        if (cos_theta > Scalar(1.0))
            cos_theta = Scalar(1.0);
        if (cos_theta < Scalar(-1.0))
            cos_theta = Scalar(-1.0);

        // Get parameters
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        Scalar K = m_K[angle_type];

        // Energy: U = (K/2) * (1 - cos_theta)
        // Split 1/3 to each particle
        Scalar energy = Scalar(0.5) * K * (Scalar(1.0) - cos_theta);
        Scalar energy_third = energy / Scalar(3.0);

        // Torque on i (lab frame): tau_i = (K/2) * cross(n_hat, d_hat)
        vec3<Scalar> tau_i = Scalar(0.5) * K * cross(n_hat, d_hat);

        // Force on j: F_j = -(K/2) / |d| * (n_hat - cos_theta * d_hat)
        // This is -dU/dr_j where U depends on r_j through d_hat
        vec3<Scalar> n_perp = n_hat - cos_theta * d_hat;
        vec3<Scalar> F_j = Scalar(-0.5) * K * d_inv * n_perp;
        vec3<Scalar> F_k = -F_j; // Newton's 3rd law: F_k = -F_j

        // Virial: W_ij = (1/2) * sum_pairs F_pair^a * dr_pair^b
        // The pair interaction is between j and k via displacement d
        // We split 1/3 of the virial to each of the 3 particles
        // virial = F_j^a * d^b  (since d = r_k - r_j and F_k = -F_j)
        Scalar virial[6];
        virial[0] = Scalar(1. / 3.) * F_j.x * d.x; // xx
        virial[1] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.y * d.x + F_j.x * d.y); // xy
        virial[2] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.z * d.x + F_j.x * d.z); // xz
        virial[3] = Scalar(1. / 3.) * F_j.y * d.y; // yy
        virial[4] = Scalar(1. / 3.) * Scalar(0.5) * (F_j.z * d.y + F_j.y * d.z); // yz
        virial[5] = Scalar(1. / 3.) * F_j.z * d.z; // zz

        // Accumulate: only for local (non-ghost) particles
        if (idx_i < m_pdata->getN())
            {
            // No force on i (the oriented particle), only torque and energy
            h_force.data[idx_i].w += energy_third;
            h_torque.data[idx_i].x += tau_i.x;
            h_torque.data[idx_i].y += tau_i.y;
            h_torque.data[idx_i].z += tau_i.z;
            for (int v = 0; v < 6; v++)
                h_virial.data[v * virial_pitch + idx_i] += virial[v];
            }

        if (idx_j < m_pdata->getN())
            {
            h_force.data[idx_j].x += F_j.x;
            h_force.data[idx_j].y += F_j.y;
            h_force.data[idx_j].z += F_j.z;
            h_force.data[idx_j].w += energy_third;
            for (int v = 0; v < 6; v++)
                h_virial.data[v * virial_pitch + idx_j] += virial[v];
            }

        if (idx_k < m_pdata->getN())
            {
            h_force.data[idx_k].x += F_k.x;
            h_force.data[idx_k].y += F_k.y;
            h_force.data[idx_k].z += F_k.z;
            h_force.data[idx_k].w += energy_third;
            for (int v = 0; v < 6; v++)
                h_virial.data[v * virial_pitch + idx_k] += virial[v];
            }
        }
    }

namespace detail
    {
void export_AlignAngleForceCompute(pybind11::module& m)
    {
    pybind11::class_<AlignAngleForceCompute,
                     ForceCompute,
                     std::shared_ptr<AlignAngleForceCompute>>(m, "AlignAngleForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &AlignAngleForceCompute::setParamsPython)
        .def("getParams", &AlignAngleForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
