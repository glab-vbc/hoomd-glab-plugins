// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SinSqDihedralForceCompute.h"
#include "SinSqDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file SinSqDihedralForceComputeGPU.h
    \brief Declares the SinSqDihedralForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SINSQDIHEDRALFORCECOMPUTEGPU_H__
#define __SINSQDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {

//! Implements the sin²-multiplied dihedral force on the GPU
class PYBIND11_EXPORT SinSqDihedralForceComputeGPU : public SinSqDihedralForceCompute
    {
    public:
    //! Constructs the compute
    SinSqDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> system);
    //! Destructor
    ~SinSqDihedralForceComputeGPU();

    //! Set the parameters
    virtual void
    setParams(unsigned int type, Scalar K, Scalar sign, int multiplicity, Scalar phi_0);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params;            //!< Parameters stored on the GPU (k, sign, m, phi_0)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_SinSqDihedralForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif
