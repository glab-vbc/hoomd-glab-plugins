"""Tests for the SoftHarmonicAngle (soft/capped harmonic) angle force."""

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
# Reference potential (mirrors SoftHarmonicTail.h, x = theta - t0)
# ---------------------------------------------------------------------------

def _U(theta, k, t0, x_c, tail):
    x = theta - t0
    ax = abs(x)
    if tail == "flat":
        if ax < x_c:
            s2 = (x / x_c) ** 2
            return 0.5 * k * x * x * (1.0 - s2 + s2 * s2 / 3.0)
        return k * x_c * x_c / 6.0
    else:  # linear
        if ax < x_c:
            return 0.5 * k * x * x
        return k * x_c * ax - 0.5 * k * x_c * x_c


def _theta(positions):
    a, b, c = [numpy.asarray(p, float) for p in positions]
    dab = a - b
    dcb = c - b
    cos = numpy.dot(dab, dcb) / (numpy.linalg.norm(dab) * numpy.linalg.norm(dcb))
    return numpy.arccos(numpy.clip(cos, -1.0, 1.0))


def _analytic_energy(positions, k, t0, x_c, tail):
    return _U(_theta(positions), k, t0, x_c, tail)


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
    soft = glab_forces.SoftHarmonicAngle()
    soft.params["A-A-A"] = params
    integrator.forces.append(soft)
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, soft


def _get_forces(sim):
    forces = sim.operations.integrator.forces[0]
    return numpy.array([forces.forces[i] for i in range(3)])


def _total_energy(sim):
    forces = sim.operations.integrator.forces[0]
    return sum(forces.energies[i] for i in range(3))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParams:
    def test_params_round_trip(self, device):
        soft = glab_forces.SoftHarmonicAngle()
        soft.params["A-A-A"] = dict(k=20.0, t0=numpy.pi, x_c=0.6, tail="linear")
        p = soft.params["A-A-A"]
        assert p["k"] == pytest.approx(20.0)
        assert p["t0"] == pytest.approx(numpy.pi)
        assert p["x_c"] == pytest.approx(0.6)
        assert p["tail"] == "linear"

    def test_default_tail_is_flat(self, device, angle_snapshot_factory):
        snap = angle_snapshot_factory(_angle_positions(100.0))
        sim, soft = _make_sim(device, snap,
                             dict(k=20.0, t0=numpy.pi / 2, x_c=0.6))
        assert soft.params["A-A-A"]["tail"] == "flat"

    def test_bad_tail_raises(self, device, angle_snapshot_factory):
        snap = angle_snapshot_factory(_angle_positions(100.0))
        with pytest.raises(Exception):
            _make_sim(device, snap,
                      dict(k=20.0, t0=numpy.pi / 2, x_c=0.6, tail="nope"))


# t0 = 90 deg, x_c = 0.6 rad (~34 deg): inside / outside, both signs.
_THETA_DEG = [90, 100, 120, 140, 70, 55, 150]


@pytest.mark.parametrize("tail", ["linear", "flat"])
@pytest.mark.parametrize("theta_deg", _THETA_DEG)
class TestEnergyAndForce:
    K, T0, XC = 25.0, numpy.pi / 2, 0.6

    def test_energy(self, device, angle_snapshot_factory, tail, theta_deg):
        params = dict(k=self.K, t0=self.T0, x_c=self.XC, tail=tail)
        positions = _angle_positions(theta_deg)
        snap = angle_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)
        e_ref = _analytic_energy(positions, **params)
        assert _total_energy(sim) == pytest.approx(e_ref, abs=E_ATOL, rel=1e-5)

    def test_forces_finite_difference(self, device, angle_snapshot_factory,
                                      tail, theta_deg):
        params = dict(k=self.K, t0=self.T0, x_c=self.XC, tail=tail)
        positions = _angle_positions(theta_deg)
        snap = angle_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)
        forces = _get_forces(sim)
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
                    f"atom={atom} dim={dim} tail={tail} theta={theta_deg}"

    def test_newtons_third_law(self, device, angle_snapshot_factory,
                               tail, theta_deg):
        params = dict(k=self.K, t0=self.T0, x_c=self.XC, tail=tail)
        snap = angle_snapshot_factory(_angle_positions(theta_deg))
        sim, _ = _make_sim(device, snap, params)
        total = _get_forces(sim).sum(axis=0)
        numpy.testing.assert_allclose(total, 0.0, atol=FD_ATOL)


class TestTailBehaviour:
    def test_flat_zero_torque_beyond_xc(self, device, angle_snapshot_factory):
        # t0 = 90 deg, x_c = 0.4 rad (~23 deg); theta = 140 deg -> well outside
        params = dict(k=25.0, t0=numpy.pi / 2, x_c=0.4, tail="flat")
        snap = angle_snapshot_factory(_angle_positions(140.0))
        sim, _ = _make_sim(device, snap, params)
        numpy.testing.assert_allclose(_get_forces(sim), 0.0, atol=FD_ATOL)
        assert _total_energy(sim) == pytest.approx(25.0 * 0.16 / 6.0,
                                                   abs=E_ATOL, rel=1e-5)

    def test_harmonic_limit_matches_builtin(self, device, angle_snapshot_factory):
        """For |theta - t0| << x_c both tails match hoomd.md.angle.Harmonic."""
        k, t0 = 25.0, numpy.pi / 2
        positions = _angle_positions(93.0)  # x ~ 0.05 rad
        for tail in ("linear", "flat"):
            snap = angle_snapshot_factory(positions)
            sim, _ = _make_sim(device, snap,
                               dict(k=k, t0=t0, x_c=1.5, tail=tail))
            e_soft = _total_energy(sim)
            f_soft = _get_forces(sim)

            snap2 = angle_snapshot_factory(positions)
            sim2 = hoomd.Simulation(device=device)
            sim2.create_state_from_snapshot(snap2)
            integ = hoomd.md.Integrator(dt=0.0)
            harm = hoomd.md.angle.Harmonic()
            harm.params["A-A-A"] = dict(k=k, t0=t0)
            integ.forces.append(harm)
            sim2.operations.integrator = integ
            sim2.run(0)
            e_harm = sum(harm.energies[i] for i in range(3))
            f_harm = numpy.array([harm.forces[i] for i in range(3)])

            rel = 1e-2 if tail == "flat" else 1e-4
            assert e_soft == pytest.approx(e_harm, rel=rel, abs=E_ATOL)
            numpy.testing.assert_allclose(f_soft, f_harm, rtol=rel, atol=FD_ATOL)
