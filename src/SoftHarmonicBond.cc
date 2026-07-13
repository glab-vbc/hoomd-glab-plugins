// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/PotentialBond.h"

#include "EvaluatorBondSoftHarmonic.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_SoftHarmonicBond(pybind11::module& m)
    {
    export_PotentialBond<EvaluatorBondSoftHarmonic>(m, "PotentialBondSoftHarmonic");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
