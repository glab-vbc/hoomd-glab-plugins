// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "SoftHarmonicAngleForceCompute.h"
#include "SoftHarmonicAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file SoftHarmonicAngleForceComputeGPU.h
    \brief Declares the SoftHarmonicAngleForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SOFTHARMONICANGLEFORCECOMPUTEGPU_H__
#define __SOFTHARMONICANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {

//! Implements the soft/capped harmonic angle force on the GPU
class PYBIND11_EXPORT SoftHarmonicAngleForceComputeGPU : public SoftHarmonicAngleForceCompute
    {
    public:
    //! Constructs the compute
    SoftHarmonicAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~SoftHarmonicAngleForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar t_0, Scalar x_c, int mode);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params;            //!< Parameters on the GPU (k, t_0, x_c, mode)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_SoftHarmonicAngleForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif
