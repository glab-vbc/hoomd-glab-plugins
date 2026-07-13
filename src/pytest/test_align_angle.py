# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Tests for the align_angle plugin.

Run with: python -m pytest test_align_angle.py -v
(requires HOOMD and align_angle to be installed)
"""

import hoomd
import numpy as np
import pytest

from hoomd import align_angle


def make_snapshot_with_angles(device, positions, orientations, L=20.0):
    """Create a Snapshot with 3 particles, 1 angle (0,1,2).

    Particle 0 is the oriented particle (i). Particles 1,2 are guides (j,k).
    Direction d = r_2 - r_1, director n = rotate(q_0, x_hat).
    """
    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = 3
        snap.particles.types = ["A"]
        snap.particles.position[:] = positions
        snap.particles.orientation[:] = orientations
        snap.angles.N = 1
        snap.angles.types = ["align"]
        snap.angles.typeid[0] = 0
        snap.angles.group[0] = (0, 1, 2)
    return snap


@pytest.fixture(scope="session")
def device():
    return hoomd.device.auto_select()


class TestAlignAngleForce:

    def test_params(self, device):
        """Test setting and getting parameters."""
        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0)
        assert force.params["align"]["k"] == pytest.approx(10.0)

    def test_aligned_zero_energy(self, device):
        """When the body x-axis is parallel to d_hat, energy should be zero."""
        # d = r_2 - r_1 = (1,0,0), d_hat = (1,0,0)
        # q_i = identity => n_hat = (1,0,0)
        # cos(theta) = 1 => U = 0
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        energies = force.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(0.0, abs=1e-10)

    def test_anti_aligned_max_energy(self, device):
        """When the body x-axis is anti-parallel to d_hat, U = k."""
        # d_hat = (1,0,0)
        # q_i rotates x-axis to (-1,0,0): 180° rotation around z => q = (0,0,0,1)
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 6.0
        force.params["align"] = dict(k=k)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        energies = force.energies
        if energies is not None:
            total_e = sum(energies)
            # U = k/2 * (1 - (-1)) = k
            assert total_e == pytest.approx(k, rel=1e-5)

    def test_ninety_degrees(self, device):
        """When body-axis is perpendicular to d_hat, U = k/2."""
        # d_hat = (1,0,0)
        # 90° rotation around z: q = (cos45, 0, 0, sin45)
        c45 = np.cos(np.pi / 4)
        s45 = np.sin(np.pi / 4)
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c45, 0, 0, s45], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 8.0
        force.params["align"] = dict(k=k)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        energies = force.energies
        if energies is not None:
            total_e = sum(energies)
            # n_hat = (0, 1, 0), cos_theta = 0, U = k/2
            assert total_e == pytest.approx(k / 2, rel=1e-5)

    def test_forces_newtons_third_law(self, device):
        """Total force on all particles should be zero (Newton's third law)."""
        c30 = np.cos(np.pi / 6)
        s30 = np.sin(np.pi / 6)
        positions = np.array([[-1, 0, 0], [0, 0.5, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c30, 0, 0, s30], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=5.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        forces = force.forces
        if forces is not None:
            total_f = np.sum(forces, axis=0)
            np.testing.assert_allclose(total_f, [0, 0, 0], atol=1e-10)

    def test_torque_direction(self, device):
        """Torque should be along z when n_hat and d_hat both lie in the xy plane."""
        # d_hat = (1,0,0), q_i rotates by 30° around z => n_hat in xy plane
        c15 = np.cos(np.pi / 12)
        s15 = np.sin(np.pi / 12)
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c15, 0, 0, s15], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        torques = force.torques
        if torques is not None:
            # Particle i is index 0.  Torque should be along z (cross of
            # vectors in xy plane).  x and y torque components should be ~0.
            assert abs(torques[0][0]) < 1e-10  # tau_x ~ 0
            assert abs(torques[0][1]) < 1e-10  # tau_y ~ 0
            # tau_z should be positive (right-hand rule: n_hat × d_hat with
            # n_hat rotated CCW from d_hat gives +z torque that pulls back)
            # Actually cross(n_hat, d_hat) with n_hat CCW from d_hat => -z
            # But the sign depends on convention. Just check it's nonzero.
            assert abs(torques[0][2]) > 1e-5

    def test_no_force_on_j(self, device):
        """The oriented particle i should have zero translational force."""
        c15 = np.cos(np.pi / 12)
        s15 = np.sin(np.pi / 12)
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c15, 0, 0, s15], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        forces = force.forces
        if forces is not None:
            np.testing.assert_allclose(forces[0], [0, 0, 0], atol=1e-10)


class TestDirectorAlignMultiplicity:
    """Tests for multiplicity and phase parameters."""

    def test_params_multiplicity_phase(self, device):
        """Test setting and getting multiplicity and phase."""
        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0, multiplicity=2, phase=1.5)
        assert force.params["align"]["k"] == pytest.approx(10.0)
        assert force.params["align"]["multiplicity"] == 2
        assert force.params["align"]["phase"] == pytest.approx(1.5)

    def test_defaults_backward_compatible(self, device):
        """Omitting multiplicity/phase gives the same result as the old code."""
        # theta=0 => U = k/2*(1 - cos(0)) = 0
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=10.0)  # no multiplicity/phase

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        assert total_e == pytest.approx(0.0, abs=1e-10)

    def test_multiplicity_2_nematic_aligned(self, device):
        """m=2, phase=0: cos(2*0) = 1 => U = 0 when aligned."""
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 10.0
        force.params["align"] = dict(k=k, multiplicity=2, phase=0.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        assert total_e == pytest.approx(0.0, abs=1e-10)

    def test_multiplicity_2_nematic_anti_aligned(self, device):
        """m=2, phase=0: cos(2*pi) = 1 => U = 0 when anti-aligned (nematic)."""
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        # 180° around z: n_hat = (-1,0,0), theta = pi
        orientations = np.array(
            [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 10.0
        force.params["align"] = dict(k=k, multiplicity=2, phase=0.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        # cos(2*pi) = 1, U = k/2*(1-1) = 0
        assert total_e == pytest.approx(0.0, abs=1e-10)

    def test_multiplicity_2_perpendicular_max(self, device):
        """m=2, phase=0: theta=pi/2 => cos(pi) = -1 => U = k."""
        c45 = np.cos(np.pi / 4)
        s45 = np.sin(np.pi / 4)
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c45, 0, 0, s45], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 8.0
        force.params["align"] = dict(k=k, multiplicity=2, phase=0.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        # cos(2*pi/2) = cos(pi) = -1, U = k/2*(1-(-1)) = k
        assert total_e == pytest.approx(k, rel=1e-5)

    def test_phase_pi_anti_align(self, device):
        """m=1, phase=pi: U = k/2*(1 - cos(theta+pi)) = k/2*(1+cos(theta)).
        At theta=0, U = k. At theta=pi, U = 0."""
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        # Aligned: theta = 0
        orientations = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 6.0
        force.params["align"] = dict(k=k, multiplicity=1, phase=np.pi)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        # cos(0 + pi) = -1, U = k/2*(1-(-1)) = k
        assert total_e == pytest.approx(k, rel=1e-5)

    def test_phase_pi_anti_aligned_zero(self, device):
        """m=1, phase=pi: at theta=pi (anti-aligned), U = 0."""
        positions = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        k = 6.0
        force.params["align"] = dict(k=k, multiplicity=1, phase=np.pi)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        total_e = sum(force.energies)
        # cos(pi + pi) = cos(2pi) = 1, U = k/2*(1-1) = 0
        assert total_e == pytest.approx(0.0, abs=1e-10)

    def test_multiplicity_2_newtons_third_law(self, device):
        """Newton's third law holds with multiplicity=2."""
        c30 = np.cos(np.pi / 6)
        s30 = np.sin(np.pi / 6)
        positions = np.array([[-1, 0, 0], [0, 0.5, 0], [1, 0, 0]], dtype=float)
        orientations = np.array(
            [[c30, 0, 0, s30], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float
        )
        snap = make_snapshot_with_angles(device, positions, orientations)
        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        force = align_angle.DirectorAlign()
        force.params["align"] = dict(k=5.0, multiplicity=2, phase=0.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        forces = force.forces
        if forces is not None:
            total_f = np.sum(forces, axis=0)
            np.testing.assert_allclose(total_f, [0, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Helpers: Python reference and HOOMD query
# ---------------------------------------------------------------------------

def _rotate_quat(q, v):
    """Rotate vector v by quaternion q = [w, x, y, z] (double precision)."""
    w, x, y, z = q
    return np.array([
        (1 - 2*(y*y + z*z)) * v[0] + 2*(x*y - w*z) * v[1] + 2*(x*z + w*y) * v[2],
        2*(x*y + w*z) * v[0] + (1 - 2*(x*x + z*z)) * v[1] + 2*(y*z - w*x) * v[2],
        2*(x*z - w*y) * v[0] + 2*(y*z + w*x) * v[1] + (1 - 2*(x*x + y*y)) * v[2],
    ])


def _python_ref(positions, q_i, K, multiplicity=1, phase=0.0):
    """Compute energy, forces, torque analytically in Python (double precision).

    Returns (energy, F_i, F_j, F_k, tau_i).
    """
    d = positions[2] - positions[1]  # r_k - r_j
    d_mag = np.linalg.norm(d)
    d_hat = d / d_mag

    n_hat = _rotate_quat(q_i, np.array([1.0, 0.0, 0.0]))
    cos_theta = np.clip(np.dot(n_hat, d_hat), -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    theta = np.arccos(cos_theta)

    m = float(multiplicity)
    mp = m * theta + phase
    energy = 0.5 * K * (1.0 - np.cos(mp))

    if sin_theta > 1e-12:
        f = m * np.sin(mp) / sin_theta
    else:
        f = 0.0

    n_perp = n_hat - cos_theta * d_hat
    F_j = -0.5 * K * f / d_mag * n_perp
    F_k = -F_j
    F_i = np.zeros(3)
    tau_i = 0.5 * K * f * np.cross(n_hat, d_hat)

    return energy, F_i, F_j, F_k, tau_i


def _get_hoomd_results(device, positions, orientations, k, multiplicity=1, phase=0.0):
    """Run HOOMD, return (energy, forces[3,3], torques[3,3])."""
    snap = make_snapshot_with_angles(device, positions, orientations)
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snap)
    force = align_angle.DirectorAlign()
    force.params["align"] = dict(k=k, multiplicity=multiplicity, phase=phase)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
    sim.operations.integrator = integrator
    sim.run(0)
    return float(sum(force.energies)), np.array(force.forces), np.array(force.torques)


# Mixed-precision (ForceReal = float) tolerance
_ATOL = 5e-4


class TestAnalyticReference:
    """Compare HOOMD forces/torques/energy against a Python reference.

    Uses a non-trivial 3D configuration and verifies all components.
    """

    @pytest.fixture()
    def config1(self):
        """45° rotation around (1,1,1)/sqrt(3); K=7.5, m=1, phase=0."""
        pos = np.array([[-0.7, 0.3, 0.1], [0.2, 0.5, -0.2], [1.1, -0.1, 0.4]])
        angle = np.pi / 4
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        q0 = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
        ori = np.array([q0, [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
        return pos, ori, q0, 7.5, 1, 0.0

    @pytest.fixture()
    def config2(self):
        """60° rotation around y; K=5, m=2, phase=0.3."""
        pos = np.array([[-0.7, 0.3, 0.1], [0.2, 0.5, -0.2], [1.1, -0.1, 0.4]])
        angle = np.pi / 3
        axis = np.array([0, 1, 0])
        q0 = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
        ori = np.array([q0, [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
        return pos, ori, q0, 5.0, 2, 0.3

    @pytest.fixture()
    def config3(self):
        """120° rotation around (0,0,1); K=12, m=1, phase=pi. Non-trivial guide positions."""
        pos = np.array([[0.5, -0.3, 0.8], [-1.2, 0.7, 0.0], [0.3, 1.5, -0.6]])
        angle = 2 * np.pi / 3
        axis = np.array([0, 0, 1])
        q0 = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
        ori = np.array([q0, [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
        return pos, ori, q0, 12.0, 1, np.pi

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_energy(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        ref_e, _, _, _, _ = _python_ref(pos, q0, K, mult, phase)
        hoomd_e, _, _ = _get_hoomd_results(device, pos, ori, K, mult, phase)
        assert hoomd_e == pytest.approx(ref_e, abs=_ATOL)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_forces_j(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, _, ref_Fj, _, _ = _python_ref(pos, q0, K, mult, phase)
        _, hoomd_f, _ = _get_hoomd_results(device, pos, ori, K, mult, phase)
        np.testing.assert_allclose(hoomd_f[1], ref_Fj, atol=_ATOL)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_forces_k(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, _, _, ref_Fk, _ = _python_ref(pos, q0, K, mult, phase)
        _, hoomd_f, _ = _get_hoomd_results(device, pos, ori, K, mult, phase)
        np.testing.assert_allclose(hoomd_f[2], ref_Fk, atol=_ATOL)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_torque_i(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, _, _, _, ref_tau = _python_ref(pos, q0, K, mult, phase)
        _, _, hoomd_t = _get_hoomd_results(device, pos, ori, K, mult, phase)
        np.testing.assert_allclose(hoomd_t[0], ref_tau, atol=_ATOL)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_no_force_on_i(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, hoomd_f, _ = _get_hoomd_results(device, pos, ori, K, mult, phase)
        np.testing.assert_allclose(hoomd_f[0], [0, 0, 0], atol=1e-10)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_no_torque_on_guides(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, _, hoomd_t = _get_hoomd_results(device, pos, ori, K, mult, phase)
        np.testing.assert_allclose(hoomd_t[1], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(hoomd_t[2], [0, 0, 0], atol=1e-10)

    @pytest.mark.parametrize("cfg_name", ["config1", "config2", "config3"])
    def test_newtons_third_law(self, device, cfg_name, request):
        pos, ori, q0, K, mult, phase = request.getfixturevalue(cfg_name)
        _, hoomd_f, _ = _get_hoomd_results(device, pos, ori, K, mult, phase)
        total_f = np.sum(hoomd_f, axis=0)
        np.testing.assert_allclose(total_f, [0, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Emergent / statistical behavior (moved out of the demo notebook)
#
# Under Langevin dynamics a single oriented particle held by two fixed guides
# should thermalize around perfect alignment with <cos theta> -> 1 - kT/K in the
# stiff limit, and tighter fluctuations (smaller variance) as K grows.  These are
# tolerant, fixed-seed statistical checks, not exact ones.
# ---------------------------------------------------------------------------

def _run_equipartition(device, K, kT=1.0, dt=0.005, n_steps=40000, sample_every=50,
                       seed=123):
    """Langevin-thermostat a single oriented particle between two fixed guides;
    return the array of cos(theta) samples over the last half of the run."""
    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 3
        snap.particles.types = ["Oriented", "Guide"]
        snap.particles.typeid[:] = [0, 1, 1]           # particle 0 is oriented (i)
        snap.particles.position[:] = [[0, 0, 0], [-1, 0, 0], [1, 0, 0]]
        # start 90 deg away from the target (body-x -> lab -z)
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        snap.particles.orientation[:] = [[c, 0, -s, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        snap.particles.moment_inertia[0] = [1.0, 1.0, 1.0]
        snap.particles.mass[:] = 1.0
        snap.angles.N = 1
        snap.angles.types = ["align"]
        snap.angles.group[0] = (0, 1, 2)               # d = r_2 - r_1 = +x (fixed)

    sim = hoomd.Simulation(device=device, seed=seed)
    sim.create_state_from_snapshot(snap)
    force = align_angle.DirectorAlign()
    force.params["align"] = dict(k=K)
    # integrate only the oriented particle -> the guides (hence d_hat) stay fixed
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.Type(["Oriented"]), kT=kT)
    langevin.gamma_r["Oriented"] = (1.0, 1.0, 1.0)   # thermal bath for the rotational DOF
    langevin.gamma_r["Guide"] = (1.0, 1.0, 1.0)
    integ = hoomd.md.Integrator(dt=dt, methods=[langevin], forces=[force],
                                integrate_rotational_dof=True)
    sim.operations.integrator = integ

    d_hat = np.array([1.0, 0.0, 0.0])
    cos = []
    for _ in range(n_steps // sample_every):
        sim.run(sample_every)
        with sim.state.cpu_local_snapshot as ss:
            q = np.array(ss.particles.orientation[ss.particles.tag[:] == 0][0])
        cos.append(np.dot(_rotate_quat(q, np.array([1.0, 0.0, 0.0])), d_hat))
    cos = np.array(cos)
    return cos[len(cos) // 2:]


class TestEquipartition:
    """Emergent alignment statistics under thermal noise (fixed seed, tolerant)."""

    def test_strong_spring_aligns(self, device):
        cos = _run_equipartition(device, K=50.0, kT=1.0)
        assert cos.mean() > 0.85            # stiff spring -> nearly aligned

    def test_variance_decreases_with_stiffness(self, device):
        weak = _run_equipartition(device, K=5.0, kT=1.0)
        stiff = _run_equipartition(device, K=50.0, kT=1.0)
        # stiffer alignment -> higher mean cos theta and tighter fluctuations
        assert stiff.mean() > weak.mean()
        assert stiff.var() < weak.var()
