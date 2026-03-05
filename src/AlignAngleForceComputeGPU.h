// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "AlignAngleForceCompute.h"
#include "AlignAngleForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file AlignAngleForceComputeGPU.h
    \brief Declares the AlignAngleForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __ALIGNANGLEFORCECOMPUTEGPU_H__
#define __ALIGNANGLEFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Implements the align-angle force calculation on the GPU
/*! AlignAngleForceComputeGPU implements the same calculations as AlignAngleForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a GPUArray of Scalar4 values (K, multiplicity, phase, 0).

    \ingroup computes
*/
class PYBIND11_EXPORT AlignAngleForceComputeGPU : public AlignAngleForceCompute
    {
    public:
    //! Constructs the compute
    AlignAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Destructor
    ~AlignAngleForceComputeGPU();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, unsigned int multiplicity, Scalar phase);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<Scalar4> m_params_gpu;        //!< Parameters on GPU (K, multiplicity, phase, 0)

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_AlignAngleForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif
