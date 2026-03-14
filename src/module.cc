// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AlignAngleForceCompute(pybind11::module& m);
void export_AnisoPotentialPairNematic(pybind11::module& m);
void export_SinSqDihedralForceCompute(pybind11::module& m);
#ifdef ENABLE_HIP
void export_AlignAngleForceComputeGPU(pybind11::module& m);
void export_AnisoPotentialPairNematicGPU(pybind11::module& m);
void export_SinSqDihedralForceComputeGPU(pybind11::module& m);
#endif
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

using namespace hoomd::md::detail;

PYBIND11_MODULE(_align_angle, m)
    {
    export_AlignAngleForceCompute(m);
    export_AnisoPotentialPairNematic(m);
    export_SinSqDihedralForceCompute(m);
#ifdef ENABLE_HIP
    export_AlignAngleForceComputeGPU(m);
    export_AnisoPotentialPairNematicGPU(m);
    export_SinSqDihedralForceComputeGPU(m);
#endif
    }
