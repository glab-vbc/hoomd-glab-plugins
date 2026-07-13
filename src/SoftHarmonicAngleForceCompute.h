// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#pragma once

#include "SoftHarmonicTail.h"
#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <stdexcept>
#include <string>

/*! \file SoftHarmonicAngleForceCompute.h
    \brief Declares a class for computing the soft/capped harmonic angle force.

    A quadratic angular well (curvature k about the equilibrium angle t0) whose
    tail either stays capped at a constant restoring torque (``tail = "linear"``)
    or releases to zero torque past a crossover deviation (``tail = "flat"``).
    See SoftHarmonicTail.h for the exact potential.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Parameters for the soft/capped harmonic angle potential
struct soft_angle_params
    {
    Scalar k;   //!< stiffness (curvature at the minimum)
    Scalar t_0; //!< equilibrium angle (radians)
    Scalar x_c; //!< crossover deviation (radians, > 0)
    int mode;   //!< SoftHarmonicMode

#ifndef __HIPCC__
    soft_angle_params() : k(0), t_0(0), x_c(1), mode(SOFT_HARMONIC_LINEAR) { }

    soft_angle_params(pybind11::dict v)
        {
        k = v["k"].cast<Scalar>();
        t_0 = v["t0"].cast<Scalar>();
        x_c = v["x_c"].cast<Scalar>();
        std::string tail = v["tail"].cast<std::string>();
        if (tail == "linear")
            mode = SOFT_HARMONIC_LINEAR;
        else if (tail == "flat")
            mode = SOFT_HARMONIC_FLAT;
        else
            throw std::invalid_argument("SoftHarmonic angle: tail must be 'linear' or 'flat'");
        if (x_c <= Scalar(0))
            throw std::invalid_argument("SoftHarmonic angle: x_c must be > 0");
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["t0"] = t_0;
        v["x_c"] = x_c;
        v["tail"] = (mode == SOFT_HARMONIC_FLAT) ? "flat" : "linear";
        return v;
        }
#endif
    } __attribute__((aligned(16)));

//! Computes soft/capped harmonic angle forces
/*! Forces are computed for every angle in the system.  The angle topology is
    accessed from SystemDefinition::getAngleData.
    \ingroup computes
*/
class PYBIND11_EXPORT SoftHarmonicAngleForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    SoftHarmonicAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~SoftHarmonicAngleForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar t_0, Scalar x_c, int mode);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a particular type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_K;   //!< K parameter for multiple angle types
    Scalar* m_t_0; //!< t_0 parameter for multiple angle types
    Scalar* m_x_c; //!< x_c crossover for multiple angle types
    int* m_mode;   //!< tail mode for multiple angle types

    std::shared_ptr<AngleData> m_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_SoftHarmonicAngleForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
