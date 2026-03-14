"""Tests for the SinSqDihedral (sin²-multiplied dihedral) force."""

import itertools
import numpy
import pytest
import hoomd
from hoomd import align_angle


def _is_mixed_precision():
    """Return True when the HOOMD build uses mixed (32-bit force) precision."""
    fp = hoomd.version.floating_point_precision
    return fp[0] != fp[1]


# Finite-difference absolute tolerance: looser for 32-bit force precision.
FD_ATOL = 0.02 if _is_mixed_precision() else 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return hoomd.device.auto_select()


@pytest.fixture(scope="session")
def dihedral_snapshot_factory(device):
    """Create a 4-particle snapshot with one dihedral (0,1,2,3).

    Parameters
    ----------
    positions : (4, 3) array-like
        Positions of the four particles.
    L : float
        Box side length.

    The dihedral type is ``"ABCD"``.
    """
    def make_snapshot(positions, L=20.0):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            snapshot.configuration.box = [L, L, L, 0, 0, 0]
            snapshot.particles.N = 4
            snapshot.particles.types = ["A"]
            snapshot.particles.position[:] = positions
            snapshot.dihedrals.N = 1
            snapshot.dihedrals.types = ["ABCD"]
            snapshot.dihedrals.typeid[0] = 0
            snapshot.dihedrals.group[0] = (0, 1, 2, 3)
        return snapshot
    return make_snapshot


def _make_sim(device, snapshot, params):
    """Build a simulation with a SinSqDihedral force and run 0 steps."""
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snapshot)
    integrator = hoomd.md.Integrator(dt=0.0)
    sinsq = align_angle.SinSqDihedral()
    sinsq.params["ABCD"] = params
    integrator.forces.append(sinsq)
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, sinsq


def _get_forces(sim):
    """Return (N, 3) force array from the simulation state."""
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        # Forces are stored in the force compute, accessible via state
        pass
    # Use the integrator force object
    forces = sim.operations.integrator.forces[0]
    return numpy.array([forces.forces[i] for i in range(4)])


def _get_energies(sim):
    """Return per-particle energies (length-4 array)."""
    forces = sim.operations.integrator.forces[0]
    return numpy.array([forces.energies[i] for i in range(4)])


def _total_energy(sim):
    """Return total potential energy."""
    return _get_energies(sim).sum()


def _analytic_energy(positions, k, d, n, phi0):
    """Compute V = (k/2)(1 + d cos(nφ − φ₀)) sin²θ₁ sin²θ₂ analytically."""
    ra, rb, rc, rd = [numpy.asarray(p, dtype=float) for p in positions]

    dab = ra - rb
    dcb = rc - rb
    ddc = rd - rc
    dcbm = -dcb

    # Cross products
    A = numpy.cross(dab, dcbm)
    B = numpy.cross(ddc, dcbm)

    raasq = numpy.dot(A, A)
    rbbsq = numpy.dot(B, B)
    rgsq = numpy.dot(dcbm, dcbm)
    dab_sq = numpy.dot(dab, dab)
    ddc_sq = numpy.dot(ddc, ddc)

    if raasq < 1e-30 or rbbsq < 1e-30:
        return 0.0

    S1 = raasq / (dab_sq * rgsq)
    S2 = rbbsq / (ddc_sq * rgsq)

    # Dihedral angle
    rg = numpy.sqrt(rgsq)
    rabinv = 1.0 / numpy.sqrt(raasq * rbbsq)
    cos_phi = numpy.clip(numpy.dot(A, B) * rabinv, -1, 1)
    sin_phi = rg * rabinv * numpy.dot(A, ddc)
    phi = numpy.arctan2(sin_phi, cos_phi)

    V0 = 0.5 * k * (1.0 + d * numpy.cos(n * phi - phi0))
    return V0 * S1 * S2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _right_angle_dihedral(phi_deg=45.0, d=1.0):
    """4 particles where θ₁ = θ₂ = 90° and dihedral angle = phi_deg.

    Central bond along x, outer bonds perpendicular in yz plane.
    """
    phi = numpy.radians(phi_deg)
    return [
        [0, d * numpy.cos(phi / 2), d * numpy.sin(phi / 2)],
        [0, 0, 0],
        [d, 0, 0],
        [d, d * numpy.cos(phi / 2), -d * numpy.sin(phi / 2)],
    ]


def _general_dihedral(theta1_deg=60.0, theta2_deg=60.0, phi_deg=45.0, d=1.0):
    """4 particles with specified bond angles and dihedral angle.

    Places b at origin, c along +x at distance d.
    a is in the xy plane making angle θ₁ at vertex b.
    d is placed so that the b-c-d angle is θ₂ and the dihedral is φ.
    """
    theta1 = numpy.radians(theta1_deg)
    theta2 = numpy.radians(theta2_deg)
    phi = numpy.radians(phi_deg)

    # b at origin, c along x
    rb = numpy.array([0.0, 0.0, 0.0])
    rc = numpy.array([d, 0.0, 0.0])

    # a makes angle θ₁ at b with bond b→c along +x
    # put a in the xy plane
    ra = rb + d * numpy.array([-numpy.cos(numpy.pi - theta1),
                                numpy.sin(numpy.pi - theta1), 0.0])

    # d makes angle θ₂ at c with bond c→b along -x
    # and dihedral φ about the b-c axis
    rd = rc + d * numpy.array([numpy.cos(numpy.pi - theta2),
                                -numpy.sin(numpy.pi - theta2) * numpy.cos(phi),
                                numpy.sin(numpy.pi - theta2) * numpy.sin(phi)])

    return [ra.tolist(), rb.tolist(), rc.tolist(), rd.tolist()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSinSqDihedralParams:
    """Parameter round-trip tests."""

    def test_params_before_attach(self, device):
        sinsq = align_angle.SinSqDihedral()
        sinsq.params["ABCD"] = dict(k=3.0, d=-1, n=2, phi0=1.5)
        p = sinsq.params["ABCD"]
        assert p["k"] == pytest.approx(3.0)
        assert p["d"] == pytest.approx(-1.0)
        assert p["n"] == 2
        assert p["phi0"] == pytest.approx(1.5)

    def test_params_after_attach(self, device, dihedral_snapshot_factory):
        positions = _right_angle_dihedral()
        snap = dihedral_snapshot_factory(positions)
        sim, sinsq = _make_sim(device, snap, dict(k=5.0, d=1, n=3, phi0=0.5))
        p = sinsq.params["ABCD"]
        assert p["k"] == pytest.approx(5.0)
        assert p["d"] == pytest.approx(1.0)
        assert p["n"] == 3
        assert p["phi0"] == pytest.approx(0.5)


class TestSinSqDihedralCollinear:
    """Tests for near-collinear geometries (the raison d'être of SinSqDihedral)."""

    def test_collinear_zero_energy(self, device, dihedral_snapshot_factory):
        """Three of four atoms exactly collinear → energy and forces = 0."""
        # a, b, c collinear along x; d off-axis
        positions = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0]]
        snap = dihedral_snapshot_factory(positions)
        sim, sinsq = _make_sim(device, snap, dict(k=10.0, d=1, n=1, phi0=0))

        assert _total_energy(sim) == pytest.approx(0.0, abs=1e-10)
        forces = _get_forces(sim)
        numpy.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_near_collinear_finite_forces(self, device, dihedral_snapshot_factory):
        """θ₁ ≈ 0.01 rad: forces should be finite and small."""
        eps = 0.01
        positions = [
            [0, eps, 0],     # a: slightly off the b-c line
            [1, 0, 0],       # b
            [2, 0, 0],       # c
            [3, 0.5, 0.5],   # d: well off-axis
        ]
        snap = dihedral_snapshot_factory(positions)
        sim, sinsq = _make_sim(device, snap, dict(k=10.0, d=1, n=1, phi0=0))

        forces = _get_forces(sim)
        # Forces should be finite (not NaN/inf) and small
        assert numpy.all(numpy.isfinite(forces))
        assert numpy.linalg.norm(forces) < 1.0  # should be tiny

    def test_near_collinear_fd(self, device, dihedral_snapshot_factory):
        """Finite-difference check near collinearity (θ₁ ≈ 5°)."""
        positions = _general_dihedral(theta1_deg=5.0, theta2_deg=60.0, phi_deg=45.0)
        params = dict(k=8.0, d=1, n=1, phi0=0)
        snap = dihedral_snapshot_factory(positions)
        sim, sinsq = _make_sim(device, snap, params)

        forces = _get_forces(sim)
        assert numpy.all(numpy.isfinite(forces))

        # Finite-difference verification
        eps = 1e-5
        for atom in range(4):
            for dim in range(3):
                pos_plus = [list(p) for p in positions]
                pos_minus = [list(p) for p in positions]
                pos_plus[atom][dim] += eps
                pos_minus[atom][dim] -= eps
                e_plus = _analytic_energy(pos_plus, **params)
                e_minus = _analytic_energy(pos_minus, **params)
                fd_force = -(e_plus - e_minus) / (2 * eps)
                assert forces[atom][dim] == pytest.approx(fd_force, abs=FD_ATOL)


class TestSinSqDihedralEnergy:
    """Energy validation tests."""

    def test_right_angle_matches_standard(self, device, dihedral_snapshot_factory):
        """When θ₁ = θ₂ = 90°, sin²θ = 1, so SinSq should match Periodic."""
        positions = _right_angle_dihedral(phi_deg=45.0)
        params_sinsq = dict(k=3.0, d=-1, n=2, phi0=numpy.pi / 2)

        # SinSqDihedral energy
        snap1 = dihedral_snapshot_factory(positions)
        sim1, _ = _make_sim(device, snap1, params_sinsq)
        e_sinsq = _total_energy(sim1)

        # Standard Periodic dihedral energy
        snap2 = dihedral_snapshot_factory(positions)
        sim2 = hoomd.Simulation(device=device)
        sim2.create_state_from_snapshot(snap2)
        integrator2 = hoomd.md.Integrator(dt=0.0)
        periodic = hoomd.md.dihedral.Periodic()
        periodic.params["ABCD"] = dict(k=3.0, d=-1, n=2, phi0=numpy.pi / 2)
        integrator2.forces.append(periodic)
        sim2.operations.integrator = integrator2
        sim2.run(0)
        e_periodic = sum(periodic.energies[i] for i in range(4))

        assert e_sinsq == pytest.approx(e_periodic, rel=1e-5)

    @pytest.mark.parametrize(
        "k, d, n, phi0, theta1, theta2, phi_dih",
        [
            (5.0, 1, 1, 0.0, 60, 60, 0),
            (5.0, 1, 1, 0.0, 60, 60, 90),
            (5.0, 1, 1, 0.0, 60, 60, 180),
            (3.0, -1, 2, numpy.pi / 4, 45, 75, 60),
            (8.0, 1, 3, numpy.pi / 6, 30, 120, 30),
            (2.0, -1, 1, 0.0, 90, 90, 45),
        ],
    )
    def test_known_energy(self, device, dihedral_snapshot_factory,
                          k, d, n, phi0, theta1, theta2, phi_dih):
        """Compare simulation energy against analytic formula."""
        positions = _general_dihedral(theta1, theta2, phi_dih)
        params = dict(k=k, d=d, n=n, phi0=phi0)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)

        e_sim = _total_energy(sim)
        e_ref = _analytic_energy(positions, k, d, n, phi0)

        tol = 1e-3 if _is_mixed_precision() else 1e-6
        assert e_sim == pytest.approx(e_ref, abs=tol)


class TestSinSqDihedralForces:
    """Force validation tests."""

    def test_newtons_third_law(self, device, dihedral_snapshot_factory):
        """Sum of all forces should be zero."""
        positions = _general_dihedral(60, 70, 45)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, dict(k=5.0, d=1, n=2, phi0=0.3))

        forces = _get_forces(sim)
        total = forces.sum(axis=0)
        numpy.testing.assert_allclose(total, 0.0, atol=1e-6)

    @pytest.mark.parametrize(
        "theta1, theta2, phi_dih",
        [
            (60, 60, 45),
            (90, 90, 90),
            (45, 75, 120),
            (30, 120, 30),
            (10, 60, 45),   # near-collinear θ₁
            (60, 170, 90),  # near-collinear θ₂
        ],
    )
    def test_forces_finite_difference(self, device, dihedral_snapshot_factory,
                                      theta1, theta2, phi_dih):
        """Compare forces against finite-difference of energy."""
        positions = _general_dihedral(theta1, theta2, phi_dih)
        params = dict(k=5.0, d=1, n=1, phi0=0)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)

        forces = _get_forces(sim)

        eps = 1e-5
        for atom in range(4):
            for dim in range(3):
                pos_plus = [list(p) for p in positions]
                pos_minus = [list(p) for p in positions]
                pos_plus[atom][dim] += eps
                pos_minus[atom][dim] -= eps
                e_plus = _analytic_energy(pos_plus, **params)
                e_minus = _analytic_energy(pos_minus, **params)
                fd_force = -(e_plus - e_minus) / (2 * eps)
                assert forces[atom][dim] == pytest.approx(fd_force, abs=FD_ATOL), \
                    f"atom={atom} dim={dim} θ₁={theta1}° θ₂={theta2}° φ={phi_dih}°"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_multiplicity(self, device, dihedral_snapshot_factory, n):
        """Test various multiplicities via energy check."""
        positions = _general_dihedral(60, 60, 45)
        params = dict(k=4.0, d=1, n=n, phi0=0)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)

        e_sim = _total_energy(sim)
        e_ref = _analytic_energy(positions, **params)
        tol = 1e-3 if _is_mixed_precision() else 1e-6
        assert e_sim == pytest.approx(e_ref, abs=tol)

    def test_phase_offset(self, device, dihedral_snapshot_factory):
        """Test non-zero phase offset."""
        positions = _general_dihedral(70, 50, 60)
        params = dict(k=6.0, d=-1, n=2, phi0=numpy.pi / 3)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)

        e_sim = _total_energy(sim)
        e_ref = _analytic_energy(positions, **params)
        tol = 1e-3 if _is_mixed_precision() else 1e-6
        assert e_sim == pytest.approx(e_ref, abs=tol)

    def test_fd_with_phase(self, device, dihedral_snapshot_factory):
        """Finite-difference check with non-zero phase."""
        positions = _general_dihedral(50, 80, 75)
        params = dict(k=4.0, d=-1, n=2, phi0=numpy.pi / 4)
        snap = dihedral_snapshot_factory(positions)
        sim, _ = _make_sim(device, snap, params)

        forces = _get_forces(sim)

        eps = 1e-5
        for atom in range(4):
            for dim in range(3):
                pos_plus = [list(p) for p in positions]
                pos_minus = [list(p) for p in positions]
                pos_plus[atom][dim] += eps
                pos_minus[atom][dim] -= eps
                e_plus = _analytic_energy(pos_plus, **params)
                e_minus = _analytic_energy(pos_minus, **params)
                fd_force = -(e_plus - e_minus) / (2 * eps)
                assert forces[atom][dim] == pytest.approx(fd_force, abs=FD_ATOL)
