// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"

#include <memory>
#include <vector>

/*! \file SinSqDihedralForceCompute.h
    \brief Declares a class for computing the sin²-multiplied dihedral force.

    The potential is:
      V = (k/2) * (1 + d * cos(n*phi - phi_0)) * sin²(theta_1) * sin²(theta_2)
    where phi is the dihedral angle and theta_1, theta_2 are the bond angles
    at the two central atoms.  The sin² prefactors smoothly cancel the
    1/sin²(theta) singularity that appears in the standard dihedral force
    decomposition, making the forces well-behaved through collinear geometries.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Parameters for the sin²-multiplied dihedral potential
struct sinsq_dihedral_params
    {
    Scalar k;      //!< Spring constant
    Scalar d;      //!< Sign factor (+1 or -1)
    int n;         //!< Multiplicity
    Scalar phi_0;  //!< Phase offset (radians)

#ifndef __HIPCC__
    sinsq_dihedral_params() : k(0.), d(0.), n(0), phi_0(0.) { }

    sinsq_dihedral_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()), n(v["n"].cast<int>()),
          phi_0(v["phi0"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["phi0"] = phi_0;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

//! Computes sin²-multiplied dihedral forces
/*! Forces are computed for every dihedral in the system.

    The dihedral topology is accessed from SystemDefinition::getDihedralData.
    \ingroup computes
*/
class PYBIND11_EXPORT SinSqDihedralForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    SinSqDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~SinSqDihedralForceCompute();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a particular type
    pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_K;     //!< K parameter for multiple dihedral types
    Scalar* m_sign;  //!< sign parameter for multiple dihedral types
    int* m_multi;    //!< multiplicity parameter for multiple dihedral types
    Scalar* m_phi_0; //!< phi_0 parameter for multiple dihedral types

    std::shared_ptr<DihedralData> m_dihedral_data; //!< Dihedral data to use in computing dihedrals

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_SinSqDihedralForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
