// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <stdexcept>
#include <string>

/*! \file CosineAngleForceCompute.h
    \brief Declares a class for computing the cosine (worm-like-chain) angle force.

    A singularity-free bending potential ``U(theta) = k (1 - cos(theta - t0))``
    with preferred angle ``t0`` (default pi = straight). Unlike the harmonic
    angle, the Cartesian force prefactor ``a = dU/d(cos theta) = -k cos t0 +
    k sin t0 (cos theta / sin theta)`` has its singular ``1/sin`` piece gated by
    ``sin t0``, so for the worm-like-chain case ``t0 in {0, pi}`` it collapses to
    the constant ``a = -/+ k`` and never diverges at collinear geometry.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Parameters for the cosine (worm-like-chain) angle potential
struct cosine_angle_params
    {
    Scalar k;   //!< stiffness (curvature at the minimum)
    Scalar t_0; //!< preferred angle (radians)

#ifndef __HIPCC__
    cosine_angle_params() : k(0), t_0(M_PI) { }

    cosine_angle_params(pybind11::dict v)
        {
        k = v["k"].cast<Scalar>();
        t_0 = v["t0"].cast<Scalar>();
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["t0"] = t_0;
        return v;
        }
#endif
    } __attribute__((aligned(16)));

//! Computes cosine (worm-like-chain) angle forces
/*! Forces are computed for every angle in the system.  The angle topology is
    accessed from SystemDefinition::getAngleData.
    \ingroup computes
*/
class PYBIND11_EXPORT CosineAngleForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    CosineAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~CosineAngleForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

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
    Scalar* m_t_0; //!< t_0 preferred angle for multiple angle types

    std::shared_ptr<AngleData> m_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_CosineAngleForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
