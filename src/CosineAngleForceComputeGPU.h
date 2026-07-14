// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "CosineAngleForceCompute.h"
#include "CosineAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file CosineAngleForceComputeGPU.h
    \brief Declares the CosineAngleForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __COSINEANGLEFORCECOMPUTEGPU_H__
#define __COSINEANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {

//! Implements the cosine (worm-like-chain) angle force on the GPU
class PYBIND11_EXPORT CosineAngleForceComputeGPU : public CosineAngleForceCompute
    {
    public:
    //! Constructs the compute
    CosineAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~CosineAngleForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params;            //!< Parameters on the GPU (k, t_0, cos t_0, sin t_0)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_CosineAngleForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif
