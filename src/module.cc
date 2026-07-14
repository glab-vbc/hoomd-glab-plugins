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
void export_ExternalPatchForceCompute(pybind11::module& m);
void export_SinSqDihedralForceCompute(pybind11::module& m);
void export_SoftHarmonicBond(pybind11::module& m);
void export_SoftHarmonicAngleForceCompute(pybind11::module& m);
void export_CosineAngleForceCompute(pybind11::module& m);
#ifdef ENABLE_HIP
void export_AlignAngleForceComputeGPU(pybind11::module& m);
void export_AnisoPotentialPairNematicGPU(pybind11::module& m);
void export_ExternalPatchForceComputeGPU(pybind11::module& m);
void export_SinSqDihedralForceComputeGPU(pybind11::module& m);
void export_SoftHarmonicBondGPU(pybind11::module& m);
void export_SoftHarmonicAngleForceComputeGPU(pybind11::module& m);
void export_CosineAngleForceComputeGPU(pybind11::module& m);
#endif
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

using namespace hoomd::md::detail;

PYBIND11_MODULE(_glab_forces, m)
    {
    export_AlignAngleForceCompute(m);
    export_AnisoPotentialPairNematic(m);
    export_ExternalPatchForceCompute(m);
    export_SinSqDihedralForceCompute(m);
    export_SoftHarmonicBond(m);
    export_SoftHarmonicAngleForceCompute(m);
    export_CosineAngleForceCompute(m);
#ifdef ENABLE_HIP
    export_AlignAngleForceComputeGPU(m);
    export_AnisoPotentialPairNematicGPU(m);
    export_ExternalPatchForceComputeGPU(m);
    export_SinSqDihedralForceComputeGPU(m);
    export_SoftHarmonicBondGPU(m);
    export_SoftHarmonicAngleForceComputeGPU(m);
    export_CosineAngleForceComputeGPU(m);
#endif
    }
