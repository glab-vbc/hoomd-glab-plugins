// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file ExternalPatchForceComputeGPU.h
    \brief Declares ExternalPatchForceComputeGPU — GPU implementation of
           the external-patch force.
*/

#ifndef __EXTERNALPATCHFORCECOMPUTEGPU_H__
#define __EXTERNALPATCHFORCECOMPUTEGPU_H__

#include "ExternalPatchForceCompute.h"
#include "ExternalPatchForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {

//! GPU implementation of ExternalPatchForceCompute
/*! Runs the same four-body patch interaction as the CPU version, but on the GPU.
    Uses atomicAdd for force accumulation (multiple threads contribute to the
    same particles via the neighbor list).

    \ingroup computes
*/
class PYBIND11_EXPORT ExternalPatchForceComputeGPU : public ExternalPatchForceCompute
    {
    public:
    //! Constructs the compute
    ExternalPatchForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<NeighborList> nlist);

    //! Destructor
    ~ExternalPatchForceComputeGPU();

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Actually compute the forces on the GPU
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_ExternalPatchForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif // __EXTERNALPATCHFORCECOMPUTEGPU_H__
