// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#ifndef __SOFT_HARMONIC_TAIL_H__
#define __SOFT_HARMONIC_TAIL_H__

/*! \file SoftHarmonicTail.h
    \brief Shared, nvcc-safe evaluation of the soft/capped harmonic potential.

    A single definition of the potential used by both the bond evaluator and the
    angle force-compute, on CPU and GPU, so the three code paths can never
    diverge.  With signed deviation ``x`` (``r - r0`` for a bond, ``theta - t0``
    for an angle), crossover scale ``x_c > 0`` and stiffness ``k``, both tail
    modes share ``U''(0) = k``:

      * ``SOFT_HARMONIC_LINEAR`` (Huber / capped): exactly harmonic for
        ``|x| < x_c`` then a constant restoring gradient ``k*x_c`` (force capped,
        never releases).
      * ``SOFT_HARMONIC_FLAT`` (compact quartic damping): gradient
        ``k*x*(1-(x/x_c)^2)^2`` decays smoothly to zero at ``x_c`` and stays zero
        beyond it (force releases). Energy plateaus at ``k*x_c^2/6``.
*/

// SOFT_HD is __device__ under nvcc, blank on the host compiler.
#ifdef __HIPCC__
#define SOFT_HD __device__
#else
#define SOFT_HD
#endif

namespace hoomd
    {
namespace md
    {
//! Tail-mode identifiers for the soft/capped harmonic potential.
enum SoftHarmonicMode
    {
    SOFT_HARMONIC_LINEAR = 0, //!< constant-gradient (Huber) tail
    SOFT_HARMONIC_FLAT = 1     //!< gradient decays to zero (quartic) tail
    };

//! Evaluate the soft/capped harmonic potential.
/*! \tparam Real Scalar type (Scalar on the CPU, ForceReal on the GPU).
    \param mode SoftHarmonicMode selecting the tail behaviour.
    \param k Stiffness (curvature at the minimum).
    \param xc Crossover deviation (> 0).
    \param x Signed deviation from the minimum.
    \param dUdx Output: gradient dU/dx.
    \param U Output: potential energy.
*/
template<typename Real>
SOFT_HD inline void
softHarmonicTail(int mode, Real k, Real xc, Real x, Real& dUdx, Real& U)
    {
    Real ax = (x < Real(0)) ? -x : x;

    if (mode == SOFT_HARMONIC_FLAT)
        {
        if (ax < xc)
            {
            Real s2 = (x * x) / (xc * xc);
            Real g = Real(1) - s2; // (1 - s^2)
            dUdx = k * x * g * g;
            U = Real(0.5) * k * x * x * (Real(1) - s2 + s2 * s2 * (Real(1) / Real(3)));
            }
        else
            {
            dUdx = Real(0);
            U = k * xc * xc * (Real(1) / Real(6)); // plateau
            }
        }
    else // SOFT_HARMONIC_LINEAR (Huber)
        {
        if (ax < xc)
            {
            dUdx = k * x;
            U = Real(0.5) * k * x * x;
            }
        else
            {
            Real sgn = (x >= Real(0)) ? Real(1) : Real(-1);
            dUdx = k * xc * sgn;
            U = k * xc * ax - Real(0.5) * k * xc * xc;
            }
        }
    }

    } // end namespace md
    } // end namespace hoomd

#endif // __SOFT_HARMONIC_TAIL_H__
