"""Tests for the CosineAngle (worm-like-chain) angle force.

Potential: U(theta) = k (1 - cos(theta - t0)), preferred angle t0 (default pi).
The force prefactor a = dU/d(cos theta) = -k cos t0 + k sin t0 (cos theta/sin theta)
has its singular 1/sin piece gated by sin t0, so t0 in {0, pi} is exactly
singularity-free (a = -/+ k constant) and the force stays bounded at collinear
geometry, unlike hoomd.md.angle.Harmonic.
"""

import numpy
import pytest
import hoomd
from hoomd import glab_forces


def _is_mixed_precision():
    fp = hoomd.version.floating_point_precision
    return fp[0] != fp[1]


FD_ATOL = 0.02 if _is_mixed_precision() else 1e-5
E_ATOL = 1e-3 if _is_mixed_precision() else 1e-7


# ---------------------------------------------------------------------------
# Reference potential:  U(theta) = k (1 - cos(theta - t0))
# ---------------------------------------------------------------------------

def _U(theta, k, t0):
    return k * (1.0 - numpy.cos(theta - t0))


def _theta(positions):
    a, b, c = [numpy.asarray(p, float) for p in positions]
    dab = a - b
    dcb = c - b
    cos = numpy.dot(dab, dcb) / (numpy.linalg.norm(dab) * numpy.linalg.norm(dcb))
    return numpy.arccos(numpy.clip(cos, -1.0, 1.0))


def _analytic_energy(positions, k, t0):
    return _U(_theta(positions), k, t0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _rot(v, axis, angle):
    axis = numpy.asarray(axis, float)
    axis /= numpy.linalg.norm(axis)
    v = numpy.asarray(v, float)
    return (v * numpy.cos(angle)
            + numpy.cross(axis, v) * numpy.sin(angle)
            + axis * numpy.dot(axis, v) * (1.0 - numpy.cos(angle)))


def _angle_positions(theta_deg):
    """Three particles a-b-c (b = vertex) subtending angle theta_deg, in a
    generic 3D orientation with unequal arm lengths (exercises the projection).
    """
    theta = numpy.radians(theta_deg)
    b = numpy.array([0.1, 0.2, -0.1])
    u1 = numpy.array([1.0, 0.3, -0.2])
    u1 /= numpy.linalg.norm(u1)
    axis = numpy.cross(u1, [0.2, 1.0, 0.5])
    u2 = _rot(u1, axis, theta)
    a = b + 1.3 * u1
    c = b + 0.9 * u2
    return [a.tolist(), b.tolist(), c.tolist()]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return hoomd.device.auto_select()


@pytest.fixture(scope="session")
def angle_snapshot_factory(device):
    """Create a 3-particle snapshot with one angle (0,1,2) of type ``A-A-A``."""
    def make_snapshot(positions, L=20.0):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            snapshot.configuration.box = [L, L, L, 0, 0, 0]
            snapshot.particles.N = 3
            snapshot.particles.types = ["A"]
            snapshot.particles.position[:] = positions
            snapshot.angles.N = 1
            snapshot.angles.types = ["A-A-A"]
            snapshot.angles.typeid[0] = 0
            snapshot.angles.group[0] = (0, 1, 2)
        return snapshot
    return make_snapshot


def _make_sim(device, snapshot, params):
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snapshot)
    integrator = hoomd.md.Integrator(dt=0.0)
    cosine = glab_forces.CosineAngle()
    cosine.params["A-A-A"] = params
    integrator.forces.append(cosine)
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, cosine


def _make_harmonic_sim(device, snapshot, k, t0):
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snapshot)
    integrator = hoomd.md.Integrator(dt=0.0)
    harm = hoomd.md.angle.Harmonic()
    harm.params["A-A-A"] = dict(k=k, t0=t0)
    integrator.forces.append(harm)
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, harm


def _get_forces(force):
    return numpy.array([force.forces[i] for i in range(3)])


def _total_energy(force):
    return sum(force.energies[i] for i in range(3))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParams:
    def test_params_round_trip(self, device):
        cosine = glab_forces.CosineAngle()
        cosine.params["A-A-A"] = dict(k=7.0, t0=2.0)
        p = cosine.params["A-A-A"]
        assert p["k"] == pytest.approx(7.0)
        assert p["t0"] == pytest.approx(2.0)

    def test_default_t0_is_pi(self, device):
        cosine = glab_forces.CosineAngle()
        cosine.params["A-A-A"] = dict(k=7.0)
        assert cosine.params["A-A-A"]["t0"] == pytest.approx(numpy.pi)


# intermediate t0 is only exercised at moderate angles (sin theta well above the
# 0.001 clamp); the extreme-angle / collinear behaviour is tested separately for
# the singularity-free t0 = pi below.
_T0_DEG = [180, 90, 120]
_THETA_DEG = [90, 100, 120, 140, 70, 55, 150]


@pytest.mark.parametrize("t0_deg", _T0_DEG)
@pytest.mark.parametrize("theta_deg", _THETA_DEG)
class TestEnergyAndForce:
    K = 25.0

    def test_energy(self, device, angle_snapshot_factory, t0_deg, theta_deg):
        params = dict(k=self.K, t0=numpy.radians(t0_deg))
        positions = _angle_positions(theta_deg)
        snap = angle_snapshot_factory(positions)
        _, cosine = _make_sim(device, snap, params)
        e_ref = _analytic_energy(positions, **params)
        assert _total_energy(cosine) == pytest.approx(e_ref, abs=E_ATOL, rel=1e-5)

    def test_forces_finite_difference(self, device, angle_snapshot_factory,
                                      t0_deg, theta_deg):
        params = dict(k=self.K, t0=numpy.radians(t0_deg))
        positions = _angle_positions(theta_deg)
        snap = angle_snapshot_factory(positions)
        _, cosine = _make_sim(device, snap, params)
        forces = _get_forces(cosine)
        assert numpy.all(numpy.isfinite(forces))

        eps = 1e-6
        for atom in range(3):
            for dim in range(3):
                pp = [list(p) for p in positions]
                pm = [list(p) for p in positions]
                pp[atom][dim] += eps
                pm[atom][dim] -= eps
                e_plus = _analytic_energy(pp, **params)
                e_minus = _analytic_energy(pm, **params)
                fd = -(e_plus - e_minus) / (2 * eps)
                assert forces[atom][dim] == pytest.approx(fd, abs=FD_ATOL), \
                    f"atom={atom} dim={dim} t0={t0_deg} theta={theta_deg}"

    def test_newtons_third_law(self, device, angle_snapshot_factory,
                               t0_deg, theta_deg):
        params = dict(k=self.K, t0=numpy.radians(t0_deg))
        snap = angle_snapshot_factory(_angle_positions(theta_deg))
        _, cosine = _make_sim(device, snap, params)
        total = _get_forces(cosine).sum(axis=0)
        numpy.testing.assert_allclose(total, 0.0, atol=FD_ATOL)


class TestCosineSpecific:
    def test_t0_pi_equals_wlc_form(self, device, angle_snapshot_factory):
        """t0 = pi reduces to the worm-like-chain form U = k (1 + cos theta)."""
        k = 12.0
        for theta_deg in (60, 90, 120, 150):
            positions = _angle_positions(theta_deg)
            snap = angle_snapshot_factory(positions)
            _, cosine = _make_sim(device, snap, dict(k=k, t0=numpy.pi))
            expected = k * (1.0 + numpy.cos(_theta(positions)))
            assert _total_energy(cosine) == pytest.approx(expected, abs=E_ATOL,
                                                          rel=1e-5)

    def test_matches_harmonic_small_deviation(self, device, angle_snapshot_factory):
        """Near the straight minimum, cosine ~ harmonic to leading order."""
        k = 25.0
        positions = _angle_positions(177.0)  # ~3 deg from straight (t0 = pi)
        snap = angle_snapshot_factory(positions)
        _, cosine = _make_sim(device, snap, dict(k=k, t0=numpy.pi))
        e_cos, f_cos = _total_energy(cosine), _get_forces(cosine)

        snap2 = angle_snapshot_factory(positions)
        _, harm = _make_harmonic_sim(device, snap2, k, numpy.pi)
        e_harm, f_harm = _total_energy(harm), _get_forces(harm)

        assert e_cos == pytest.approx(e_harm, rel=1e-2, abs=E_ATOL)
        numpy.testing.assert_allclose(f_cos, f_harm, rtol=3e-2, atol=FD_ATOL)

    def test_singularity_free_near_fold(self, device, angle_snapshot_factory):
        """At t0 = pi the force stays bounded (~k) toward a fold (theta -> 0),
        where the harmonic angle instead diverges as 1/sin(theta)."""
        k = 25.0
        positions = _angle_positions(5.0)  # near a fold
        snap = angle_snapshot_factory(positions)
        _, cosine = _make_sim(device, snap, dict(k=k, t0=numpy.pi))
        f_cos = _get_forces(cosine)
        assert numpy.all(numpy.isfinite(f_cos))
        maxF_cos = numpy.abs(f_cos).max()

        snap2 = angle_snapshot_factory(positions)
        _, harm = _make_harmonic_sim(device, snap2, k, numpy.pi)
        maxF_harm = numpy.abs(_get_forces(harm)).max()

        # cosine stays O(k); harmonic blows up by ~1/sin(5 deg) ~ 10x
        assert maxF_cos < 6.0 * k
        assert maxF_cos < 0.25 * maxF_harm
