// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/PotentialBondGPU.h"

#include "EvaluatorBondSoftHarmonic.h"

namespace hoomd
    {
namespace md
    {
// The CPU class is instantiated in SoftHarmonicBond.cc; avoid re-instantiating here.
extern template class PotentialBond<EvaluatorBondSoftHarmonic, BondData>;

namespace detail
    {

void export_SoftHarmonicBondGPU(pybind11::module& m)
    {
    export_PotentialBondGPU<EvaluatorBondSoftHarmonic>(m, "PotentialBondSoftHarmonicGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
