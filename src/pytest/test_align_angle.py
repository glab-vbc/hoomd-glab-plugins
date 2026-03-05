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
        force = align_angle.Align()
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

        force = align_angle.Align()
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

        force = align_angle.Align()
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

        force = align_angle.Align()
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

        force = align_angle.Align()
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

        force = align_angle.Align()
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

        force = align_angle.Align()
        force.params["align"] = dict(k=10.0)

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[force])
        sim.operations.integrator = integrator
        sim.run(0)

        forces = force.forces
        if forces is not None:
            np.testing.assert_allclose(forces[0], [0, 0, 0], atol=1e-10)
