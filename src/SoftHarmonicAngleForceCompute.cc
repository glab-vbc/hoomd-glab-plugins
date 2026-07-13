// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SoftHarmonicAngleForceCompute.h"

#include <math.h>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file SoftHarmonicAngleForceCompute.cc
    \brief Contains code for the SoftHarmonicAngleForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
SoftHarmonicAngleForceCompute::SoftHarmonicAngleForceCompute(
    std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_K(nullptr), m_t_0(nullptr), m_x_c(nullptr), m_mode(nullptr)
    {
    m_exec_conf->msg->notice(5) << "Constructing SoftHarmonicAngleForceCompute" << endl;

    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_angle_data->getNTypes() == 0)
        {
        throw runtime_error("No angle types in the system.");
        }

    // allocate the parameters
    m_K = new Scalar[m_angle_data->getNTypes()];
    m_t_0 = new Scalar[m_angle_data->getNTypes()];
    m_x_c = new Scalar[m_angle_data->getNTypes()];
    m_mode = new int[m_angle_data->getNTypes()];
    }

SoftHarmonicAngleForceCompute::~SoftHarmonicAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying SoftHarmonicAngleForceCompute" << endl;

    delete[] m_K;
    delete[] m_t_0;
    delete[] m_x_c;
    delete[] m_mode;
    m_K = nullptr;
    m_t_0 = nullptr;
    m_x_c = nullptr;
    m_mode = nullptr;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter
    \param t_0 Equilibrium angle in radians
    \param x_c Crossover deviation in radians (> 0)
    \param mode Tail mode (SoftHarmonicMode)

    Sets parameters for the potential of a particular angle type
*/
void SoftHarmonicAngleForceCompute::setParams(unsigned int type,
                                              Scalar K,
                                              Scalar t_0,
                                              Scalar x_c,
                                              int mode)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }
    if (x_c <= Scalar(0))
        {
        throw runtime_error("SoftHarmonic angle: x_c must be > 0");
        }

    m_K[type] = K;
    m_t_0[type] = t_0;
    m_x_c[type] = x_c;
    m_mode[type] = mode;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "angle.SoftHarmonic: specified K <= 0" << endl;
    }

void SoftHarmonicAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_angle_data->getTypeByName(type);
    auto _params = soft_angle_params(params);
    setParams(typ, _params.k, _params.t_0, _params.x_c, _params.mode);
    }

pybind11::dict SoftHarmonicAngleForceCompute::getParams(std::string type)
    {
    auto typ = m_angle_data->getTypeByName(type);
    if (typ >= m_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }
    pybind11::dict params;
    params["k"] = m_K[typ];
    params["t0"] = m_t_0[typ];
    params["x_c"] = m_x_c[typ];
    params["tail"] = (m_mode[typ] == SOFT_HARMONIC_FLAT) ? "flat" : "linear";
    return params;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void SoftHarmonicAngleForceCompute::computeForces(uint64_t timestep)
    {
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<ForceReal> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    m_force.zeroFill();
    m_virial.zeroFill();

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[2] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];
        unsigned int idx_c = h_rtag.data[angle.tag[2]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error()
                << "angle.SoftHarmonic: angle " << angle.tag[0] << " " << angle.tag[1] << " "
                << angle.tag[2] << " incomplete." << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        // calculate the two bond vectors dab and dcb (b is the vertex)
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        // apply minimum image conventions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z;
        Scalar rcb = sqrt(rsqcb);

        Scalar c_abbc = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z;
        c_abbc /= rab * rcb;

        if (c_abbc > 1.0)
            c_abbc = 1.0;
        if (c_abbc < -1.0)
            c_abbc = -1.0;

        Scalar s_abbc = sqrt(1.0 - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = 1.0 / s_abbc;

        // evaluate the potential for this angle type
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        Scalar dth = acos(c_abbc) - m_t_0[angle_type];
        Scalar dUdth, U;
        softHarmonicTail<Scalar>(m_mode[angle_type],
                                 m_K[angle_type],
                                 m_x_c[angle_type],
                                 dth,
                                 dUdth,
                                 U);

        Scalar a = -Scalar(1.0) * dUdth * s_abbc;
        Scalar a11 = a * c_abbc / rsqab;
        Scalar a12 = -a / (rab * rcb);
        Scalar a22 = a * c_abbc / rsqcb;

        Scalar fab[3], fcb[3];

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        // 1/3 of the energy to each of the three atoms in the angle
        Scalar angle_eng = U * Scalar(1.0 / 3.0);

        // upper triangular version of virial tensor, 1/3 to each atom
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        // Apply the force to each atom, accumulate energy/virial (skip ghosts)
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += fab[0];
            h_force.data[idx_a].y += fab[1];
            h_force.data[idx_a].z += fab[2];
            h_force.data[idx_a].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += angle_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= fab[0] + fcb[0];
            h_force.data[idx_b].y -= fab[1] + fcb[1];
            h_force.data[idx_b].z -= fab[2] + fcb[2];
            h_force.data[idx_b].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += angle_virial[j];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += fcb[0];
            h_force.data[idx_c].y += fcb[1];
            h_force.data[idx_c].z += fcb[2];
            h_force.data[idx_c].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_c] += angle_virial[j];
            }
        }
    }

namespace detail
    {
void export_SoftHarmonicAngleForceCompute(pybind11::module& m)
    {
    pybind11::class_<SoftHarmonicAngleForceCompute,
                     ForceCompute,
                     std::shared_ptr<SoftHarmonicAngleForceCompute>>(
        m,
        "SoftHarmonicAngleForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &SoftHarmonicAngleForceCompute::setParamsPython)
        .def("getParams", &SoftHarmonicAngleForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
