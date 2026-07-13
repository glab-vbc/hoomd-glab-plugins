// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_SOFT_HARMONIC_H__
#define __BOND_EVALUATOR_SOFT_HARMONIC_H__

#ifndef __HIPCC__
#include <stdexcept>
#include <string>
#endif

#include "MixedPrecisionCompat.h"
#include "SoftHarmonicTail.h"
#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorBondSoftHarmonic.h
    \brief Defines the bond evaluator class for the soft/capped harmonic potential.

    A harmonic core (curvature k at the minimum) whose tail either

      * ``tail = "linear"`` (mode 0, Huber): stays exactly harmonic up to a
        crossover deviation ``x_c`` and then applies a constant restoring force
        ``k * x_c`` (the bond is capped but never breaks), or
      * ``tail = "flat"`` (mode 1, compact-support quartic damping): the
        restoring force ``-k*x*(1-(x/x_c)^2)^2`` decays smoothly to zero at
        ``x_c`` and stays zero beyond it (the bond softly releases).

    With ``x = r - r0`` the two modes share ``U''(0) = k``, so ``k`` keeps its
    usual harmonic meaning and switching ``tail`` does not change the
    small-deformation physics.
*/

// DEVICE is __device__ when compiled by nvcc, blank on the host compiler.
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
struct softbond_params
    {
    ForceReal k;   //!< stiffness (curvature at the minimum)
    ForceReal r_0; //!< rest length
    ForceReal x_c; //!< crossover deviation (> 0)
    int mode;      //!< SoftHarmonicMode

#ifndef __HIPCC__
    softbond_params() : k(0), r_0(0), x_c(1), mode(SOFT_HARMONIC_LINEAR) { }

    softbond_params(ForceReal k, ForceReal r_0, ForceReal x_c, int mode)
        : k(k), r_0(r_0), x_c(x_c), mode(mode)
        {
        }

    softbond_params(pybind11::dict v)
        {
        k = v["k"].cast<ForceReal>();
        r_0 = v["r0"].cast<ForceReal>();
        x_c = v["x_c"].cast<ForceReal>();
        std::string tail = v["tail"].cast<std::string>();
        if (tail == "linear")
            mode = SOFT_HARMONIC_LINEAR;
        else if (tail == "flat")
            mode = SOFT_HARMONIC_FLAT;
        else
            throw std::invalid_argument("SoftHarmonic bond: tail must be 'linear' or 'flat'");
        if (x_c <= ForceReal(0))
            throw std::invalid_argument("SoftHarmonic bond: x_c must be > 0");
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["r0"] = r_0;
        v["x_c"] = x_c;
        v["tail"] = (mode == SOFT_HARMONIC_FLAT) ? "flat" : "linear";
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Class for evaluating the soft/capped harmonic bond potential.
class EvaluatorBondSoftHarmonic
    {
    public:
    typedef softbond_params param_type;

    //! Constructs the bond potential evaluator.
    /*! \param _rsq Squared distance between the particles.
        \param _params Per type parameters of this potential.
    */
    DEVICE EvaluatorBondSoftHarmonic(ForceReal _rsq, const param_type& _params)
        : rsq(_rsq), K(_params.k), r_0(_params.r_0), x_c(_params.x_c), mode(_params.mode)
        {
        }

    //! This potential does not use charge.
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values (unused).
    DEVICE void setCharge(ForceReal qa, ForceReal qb) { }

    //! Evaluate the force and energy.
    /*! \param force_divr Output: computed force divided by r.
        \param bond_eng Output: computed bond energy (full, per bond).
        \return True (energy is always defined).
    */
    DEVICE bool evalForceAndEnergy(ForceReal& force_divr, ForceReal& bond_eng)
        {
        ForceReal r = sqrt(rsq);
        ForceReal x = r - r_0; // signed deviation
        ForceReal dUdr;        // dU/dr (= dU/dx)

        softHarmonicTail<ForceReal>(mode, K, x_c, x, dUdr, bond_eng);

        force_divr = -dUdr / r;

// if the result is not finite (e.g. r == 0), set the force to 0
#ifdef __HIPCC__
        if (!isfinite(force_divr))
#else
        if (!std::isfinite(force_divr))
#endif
            {
            force_divr = ForceReal(0);
            }

        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential.
    static std::string getName()
        {
        return std::string("soft_harmonic");
        }
#endif

    protected:
    ForceReal rsq; //!< Stored rsq from the constructor
    ForceReal K;   //!< stiffness
    ForceReal r_0; //!< rest length
    ForceReal x_c; //!< crossover deviation
    int mode;      //!< tail mode
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_SOFT_HARMONIC_H__
