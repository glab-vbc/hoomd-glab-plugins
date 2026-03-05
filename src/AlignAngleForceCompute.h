// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/VectorMath.h"

#include <memory>
#include <vector>

/*! \file AlignAngleForceCompute.h
    \brief Declares a class for computing orientation-alignment angle forces.

    Uses HOOMD angle topology (i, j, k) where:
      - i is the oriented particle (has a quaternion orientation)
      - j, k are guide particles defining a target direction d_hat = (r_k - r_j) / |r_k - r_j|
    The potential aligns i's body-frame x-axis with d_hat:
      U = (k/2) * (1 - n_hat . d_hat)
    where n_hat = rotate(q_i, (1,0,0))
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Parameters for the align angle potential
struct align_angle_params
    {
    Scalar k; //!< Spring constant

#ifndef __HIPCC__
    align_angle_params() : k(0) { }

    align_angle_params(pybind11::dict params) : k(params["k"].cast<Scalar>()) { }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

//! Computes orientation-alignment angle forces
/*! For each angle (i, j, k), particle j aligns its body-frame x-axis
    to the direction from i to k.

    \ingroup computes
*/
class PYBIND11_EXPORT AlignAngleForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    AlignAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~AlignAngleForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this force
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags[comm_flag::orientation] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_K; //!< K parameter for multiple angle types

    std::shared_ptr<AngleData> m_angle_data; //!< Angle data to use in computing angles

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_AlignAngleForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
