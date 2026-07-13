"""Tests for the SoftHarmonic (soft/capped harmonic) bond force."""

import numpy
import pytest
import hoomd
from hoomd import align_angle


def _is_mixed_precision():
    """Return True when the HOOMD build uses mixed (32-bit force) precision."""
    fp = hoomd.version.floating_point_precision
    return fp[0] != fp[1]


# Finite-difference / energy absolute tolerance: looser for 32-bit force precision.
FD_ATOL = 0.02 if _is_mixed_precision() else 1e-5
E_ATOL = 1e-3 if _is_mixed_precision() else 1e-7


# ---------------------------------------------------------------------------
# Reference potential (mirrors SoftHarmonicTail.h)
# ---------------------------------------------------------------------------

def _U(r, k, r0, x_c, tail):
    """Analytic soft/capped harmonic bond energy at separation r."""
    x = r - r0
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


def _analytic_energy(positions, k, r0, x_c, tail):
    p0, p1 = [numpy.asarray(p, float) for p in positions]
    r = numpy.linalg.norm(p1 - p0)
    return _U(r, k, r0, x_c, tail)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return hoomd.device.auto_select()


@pytest.fixture(scope="session")
def bond_snapshot_factory(device):
    """Create a 2-particle snapshot with one bond (0,1) of type ``A-A``."""
    def make_snapshot(positions, L=20.0):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            snapshot.configuration.box = [L, L, L, 0, 0, 0]
            snapshot.particles.N = 2
            snapshot.particles.types = ["A"]
            snapshot.particles.position[:] = positions
            snapshot.bonds.N = 1
            snapshot.bonds.types = ["A-A"]
            snapshot.bonds.typeid[0] = 0
            snapshot.bonds.group[0] = (0, 1)
        return snapshot
    return make_snapshot


def _bond_positions(r, direction=(1.0, 0.7, -0.4)):
    """Two particles separated by r along a generic (non-axis) direction."""
    d = numpy.asarray(direction, float)
    d /= numpy.linalg.norm(d)
    return [[0.0, 0.0, 0.0], (r * d).tolist()]


def _make_sim(device, snapshot, params):
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snapshot)
    integrator = hoomd.md.Integrator(dt=0.0)
    soft = align_angle.SoftHarmonic()
    soft.params["A-A"] = params
    integrator.forces.append(soft)
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, soft


def _get_forces(sim):
    forces = sim.operations.integrator.forces[0]
    return numpy.array([forces.forces[i] for i in range(2)])


def _total_energy(sim):
    forces = sim.operations.integrator.forces[0]
    return sum(forces.energies[i] for i in range(2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParams:
    def test_params_round_trip(self, device):
        soft = align_angle.SoftHarmonic()
        soft.params["A-A"] = dict(k=100.0, r0=1.0, x_c=0.5, tail="flat")
        p = soft.params["A-A"]
        assert p["k"] == pytest.approx(100.0)
        assert p["r0"] == pytest.approx(1.0)
        assert p["x_c"] == pytest.approx(0.5)
        assert p["tail"] == "flat"

    def test_default_tail_is_linear(self, device, bond_snapshot_factory):
        snap = bond_snapshot_factory(_bond_positions(1.2))
        sim, soft = _make_sim(device, snap, dict(k=10.0, r0=1.0, x_c=0.5))
        assert soft.params["A-A"]["tail"] == "linear"

    def test_bad_tail_raises(self, device, bond_snapshot_factory):
        snap = bond_snapshot_factory(_bond_positions(1.2))
        with pytest.raises(Exception):
            _make_sim(device, snap, dict(k=10.0, r0=1.0, x_c=0.5, tail="bogus"))

    def test_bad_xc_raises(self, device, bond_snapshot_factory):
        snap = bond_snapshot_factory(_bond_positions(1.2))
        with pytest.raises(Exception):
            _make_sim(device, snap, dict(k=10.0, r0=1.0, x_c=-0.5, tail="linear"))


# r0 = 1.0, x_c = 0.5 -> regimes: inside (|x|<0.5), boundary, outside; both signs.
_R_VALUES = [1.05, 1.2, 1.49, 1.5, 1.7, 1.95, 0.9, 0.6, 0.55, 0.4, 0.2]


@pytest.mark.parametrize("tail", ["linear", "flat"])
@pytest.mark.parametrize("r", _R_VALUES)
class TestEnergyAndForce:
    K, R0, XC = 40.0, 1.0, 0.5

    def test_energy(self, device, bond_snapshot_factory, tail, r):
        params = dict(k=self.K, r0=self.R0, x_c=self.XC, tail=tail)
        positions = _bond_positions(r)
        snap = bond_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)
        e_ref = _analytic_energy(positions, **params)
        assert _total_energy(sim) == pytest.approx(e_ref, abs=E_ATOL, rel=1e-5)

    def test_forces_finite_difference(self, device, bond_snapshot_factory, tail, r):
        params = dict(k=self.K, r0=self.R0, x_c=self.XC, tail=tail)
        positions = _bond_positions(r)
        snap = bond_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)
        forces = _get_forces(sim)
        assert numpy.all(numpy.isfinite(forces))

        eps = 1e-5
        for atom in range(2):
            for dim in range(3):
                pp = [list(p) for p in positions]
                pm = [list(p) for p in positions]
                pp[atom][dim] += eps
                pm[atom][dim] -= eps
                e_plus = _analytic_energy(pp, **params)
                e_minus = _analytic_energy(pm, **params)
                fd = -(e_plus - e_minus) / (2 * eps)
                assert forces[atom][dim] == pytest.approx(fd, abs=FD_ATOL), \
                    f"atom={atom} dim={dim} tail={tail} r={r}"

    def test_newtons_third_law(self, device, bond_snapshot_factory, tail, r):
        params = dict(k=self.K, r0=self.R0, x_c=self.XC, tail=tail)
        snap = bond_snapshot_factory(_bond_positions(r))
        sim, _ = _make_sim(device, snap, params)
        total = _get_forces(sim).sum(axis=0)
        numpy.testing.assert_allclose(total, 0.0, atol=FD_ATOL)


class TestTailBehaviour:
    """Distinguishing behaviour of the two tails past the crossover."""

    def test_flat_zero_force_beyond_xc(self, device, bond_snapshot_factory):
        params = dict(k=40.0, r0=1.0, x_c=0.5, tail="flat")
        snap = bond_snapshot_factory(_bond_positions(1.8))  # x = 0.8 > x_c
        sim, _ = _make_sim(device, snap, params)
        forces = _get_forces(sim)
        numpy.testing.assert_allclose(forces, 0.0, atol=FD_ATOL)
        # energy is the plateau k*x_c^2/6
        assert _total_energy(sim) == pytest.approx(40.0 * 0.25 / 6.0,
                                                   abs=E_ATOL, rel=1e-5)

    def test_linear_constant_force_beyond_xc(self, device, bond_snapshot_factory):
        params = dict(k=40.0, r0=1.0, x_c=0.5, tail="linear")
        fcap = 40.0 * 0.5  # k * x_c
        for r in (1.7, 1.9):
            snap = bond_snapshot_factory(_bond_positions(r))
            sim, _ = _make_sim(device, snap, params)
            fmag = numpy.linalg.norm(_get_forces(sim)[1])
            assert fmag == pytest.approx(fcap, abs=FD_ATOL, rel=1e-5)

    def test_harmonic_limit_matches_builtin(self, device, bond_snapshot_factory):
        """For |x| << x_c both tails match hoomd.md.bond.Harmonic."""
        k, r0 = 40.0, 1.0
        positions = _bond_positions(1.05)  # x = 0.05 << x_c
        for tail in ("linear", "flat"):
            snap = bond_snapshot_factory(positions)
            sim, _ = _make_sim(device, snap,
                               dict(k=k, r0=r0, x_c=1.0, tail=tail))
            e_soft = _total_energy(sim)
            f_soft = _get_forces(sim)

            snap2 = bond_snapshot_factory(positions)
            sim2 = hoomd.Simulation(device=device)
            sim2.create_state_from_snapshot(snap2)
            integ = hoomd.md.Integrator(dt=0.0)
            harm = hoomd.md.bond.Harmonic()
            harm.params["A-A"] = dict(k=k, r0=r0)
            integ.forces.append(harm)
            sim2.operations.integrator = integ
            sim2.run(0)
            e_harm = sum(harm.energies[i] for i in range(2))
            f_harm = numpy.array([harm.forces[i] for i in range(2)])

            # flat tail has O((x/x_c)^2) corrections; loosen slightly
            rel = 1e-2 if tail == "flat" else 1e-4
            assert e_soft == pytest.approx(e_harm, rel=rel, abs=E_ATOL)
            numpy.testing.assert_allclose(f_soft, f_harm, rtol=rel, atol=FD_ATOL)
