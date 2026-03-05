# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Tests for the DirectorPair anisotropic pair potential.

Potential: U = -epsilon * cos(m * alpha + phase) * (1 - r^2/r_c^2)^2

   m = 2 (nematic): parallel and anti-parallel equally favoured
   m = 1 (polar):   only parallel favoured

Run with: python -m pytest test_nematic_pair.py -v
"""

import hoomd
import numpy as np
import pytest

from hoomd import align_angle


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (s, vx, vy, vz)."""
    s, qv = q[0], np.array(q[1:])
    return v + 2 * s * np.cross(qv, v) + 2 * np.cross(qv, np.cross(qv, v))


def make_pair_snapshot(device, pos_i, pos_j, quat_i, quat_j, L=20.0):
    """Create a snapshot with 2 particles, no bonds/angles."""
    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = 2
        snap.particles.types = ["A"]
        snap.particles.position[0] = pos_i
        snap.particles.position[1] = pos_j
        snap.particles.orientation[0] = quat_i
        snap.particles.orientation[1] = quat_j
        snap.particles.moment_inertia[:] = [1.0, 1.0, 1.0]
        snap.particles.mass[:] = 1.0
    return snap


def make_sim_with_nematic(device, snap, epsilon, r_cut, multiplicity=2, phase=0.0):
    """Set up a simulation with only the DirectorPair force."""
    sim = hoomd.Simulation(device=device)
    sim.create_state_from_snapshot(snap)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    nematic = align_angle.DirectorPair(nlist=nlist, default_r_cut=r_cut)
    nematic.params[("A", "A")] = dict(epsilon=epsilon, multiplicity=multiplicity, phase=phase)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[nematic])
    integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator
    sim.run(0)
    return sim, nematic


@pytest.fixture(scope="session")
def device():
    return hoomd.device.auto_select()


class TestDirectorPairEnergy:

    def test_parallel_aligned(self, device):
        """Parallel orientations: U = -epsilon * g(r)."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5  # within cutoff
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = -epsilon * 1.0 * g  # (n_i.n_j)^2 = 1

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],  # body x → lab x
            quat_j=[1, 0, 0, 0],  # body x → lab x
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)

    def test_anti_parallel(self, device):
        """Anti-parallel: (n_i.n_j)^2 = 1, same energy as parallel (nematic)."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = -epsilon * 1.0 * g

        # 180° around z: q = (0, 0, 0, 1) → body x → lab -x
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[0, 0, 0, 1],  # n_j = (-1,0,0)
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)

    def test_perpendicular_max_energy(self, device):
        """Perpendicular orientations with m=2: cos(pi) = -1, U = epsilon * g."""
        epsilon = 10.0
        r_cut = 3.0
        r = 1.0
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = epsilon * g  # -epsilon * cos(pi) * g = epsilon * g

        # 90° around z: q = (cos45, 0, 0, sin45) → body x → lab y
        c45 = np.cos(np.pi / 4)
        s45 = np.sin(np.pi / 4)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],       # n_i = (1,0,0)
            quat_j=[c45, 0, 0, s45],   # n_j = (0,1,0)
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)

    def test_beyond_cutoff(self, device):
        """Particles beyond cutoff: zero energy/force/torque."""
        epsilon = 10.0
        r_cut = 3.0
        r = 4.0  # beyond cutoff

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(0.0, abs=1e-10)

        forces = nematic.forces
        if forces is not None:
            np.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_at_cutoff(self, device):
        """At r = r_cut, U should be zero (envelope = 0)."""
        epsilon = 5.0
        r_cut = 3.0
        r = r_cut - 1e-6  # just inside cutoff

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert abs(total_e) < 1e-6  # very close to zero at cutoff


class TestDirectorPairForces:

    def test_force_attractive_when_aligned(self, device):
        """Aligned particles within cutoff → attractive force (toward each other)."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        forces = nematic.forces
        if forces is not None:
            # Force on i should point toward j (+x), i.e., attractive
            assert forces[0][0] > 0, "Force on i should be toward j (+x)"
            # Force on j should point toward i (-x)
            assert forces[1][0] < 0, "Force on j should be toward i (-x)"

    def test_newtons_third_law(self, device):
        """Total force and total torque should be zero."""
        epsilon = 5.0
        r_cut = 3.0
        r = 2.0

        # Non-trivial orientation: 45° angle between axes
        c = np.cos(np.pi / 8)
        s = np.sin(np.pi / 8)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0.5, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],  # 45° around z
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        forces = nematic.forces
        if forces is not None:
            total_f = np.sum(forces, axis=0)
            np.testing.assert_allclose(total_f, [0, 0, 0], atol=1e-10)

        torques = nematic.torques
        if torques is not None:
            total_t = np.sum(torques, axis=0)
            np.testing.assert_allclose(total_t, [0, 0, 0], atol=1e-10)

    def test_repulsive_force_when_perpendicular(self, device):
        """Perpendicular orientations with m=2: cos(2*pi/2) = -1 → repulsive."""
        epsilon = 10.0
        r_cut = 3.0
        r = 1.5

        c45 = np.cos(np.pi / 4)
        s45 = np.sin(np.pi / 4)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[c45, 0, 0, s45],  # n_j = (0,1,0)
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        forces = nematic.forces
        if forces is not None:
            # Force on i should point away from j (-x), i.e., repulsive
            assert forces[0][0] < 0, "Force on i should be away from j (-x)"
            # Force on j should point away from i (+x)
            assert forces[1][0] > 0, "Force on j should be away from i (+x)"
            # Newton's third law
            np.testing.assert_allclose(
                np.sum(forces, axis=0), [0, 0, 0], atol=1e-10
            )


class TestDirectorPairTorques:

    def test_no_torque_when_aligned(self, device):
        """Parallel orientations: d(n_i.n_j)^2 / d_angle = 0 → zero torque."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        torques = nematic.torques
        if torques is not None:
            np.testing.assert_allclose(torques, 0.0, atol=1e-10)

    def test_no_torque_when_anti_aligned(self, device):
        """Anti-parallel: same as parallel, zero torque."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[0, 0, 0, 1],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        torques = nematic.torques
        if torques is not None:
            np.testing.assert_allclose(torques, 0.0, atol=1e-10)

    def test_torque_drives_alignment(self, device):
        """With misaligned orientations, torque should drive toward alignment.

        After one step, (n_i . n_j)^2 should increase (orientations become
        more parallel or anti-parallel).
        """
        epsilon = 50.0
        r_cut = 5.0
        r = 1.5

        # 45° misalignment around z
        c = np.cos(np.pi / 8)
        s = np.sin(np.pi / 8)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],
        )

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        nematic = align_angle.DirectorPair(nlist=nlist, default_r_cut=r_cut)
        nematic.params[("A", "A")] = dict(epsilon=epsilon, multiplicity=2)

        # Fix positions, only integrate orientational DOF
        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.0)
        integrator = hoomd.md.Integrator(
            dt=0.001, methods=[langevin], forces=[nematic]
        )
        integrator.integrate_rotational_dof = True
        sim.operations.integrator = integrator
        sim.run(0)

        # Compute initial (n_i.n_j)^2
        with sim.state.cpu_local_snapshot as ss:
            tags = np.array(ss.particles.tag[:])
            quats = np.array(ss.particles.orientation[:])
        order = np.argsort(tags)
        quats = quats[order]
        n0 = quat_rotate(quats[0], [1, 0, 0])
        n1 = quat_rotate(quats[1], [1, 0, 0])
        cos2_before = np.dot(n0, n1) ** 2

        # Run a few steps
        sim.run(100)

        with sim.state.cpu_local_snapshot as ss:
            tags = np.array(ss.particles.tag[:])
            quats = np.array(ss.particles.orientation[:])
        order = np.argsort(tags)
        quats = quats[order]
        n0 = quat_rotate(quats[0], [1, 0, 0])
        n1 = quat_rotate(quats[1], [1, 0, 0])
        cos2_after = np.dot(n0, n1) ** 2

        assert cos2_after > cos2_before, (
            f"(n_i.n_j)^2 should increase: {cos2_before:.4f} → {cos2_after:.4f}"
        )


class TestDirectorPairNumerical:

    def test_energy_formula(self, device):
        """Verify energy matches the analytic formula for a general configuration."""
        epsilon = 7.0
        r_cut = 4.0

        # q_j: 30° rotation around z
        theta = np.pi / 6
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        r_vec = np.array([1.0, 0.8, 0.3])
        r = np.linalg.norm(r_vec)

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=r_vec.tolist(),
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut)

        # Analytic
        n_i = np.array([1, 0, 0], dtype=float)
        n_j = quat_rotate([c, 0, 0, s], np.array([1, 0, 0], dtype=float))
        cos_nij = np.dot(n_i, n_j)
        alpha = np.arccos(cos_nij)
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected = -epsilon * np.cos(2 * alpha) * g

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected, rel=1e-5)

    def test_force_finite_difference(self, device):
        """Compare force with numerical gradient of energy."""
        epsilon = 5.0
        r_cut = 3.5
        h = 1e-5  # finite difference step

        r_base = np.array([1.2, 0.7, 0.4])

        # 60° rotation around y for particle j
        theta = np.pi / 3
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        qj = [c, 0, s, 0]

        snap0 = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=r_base.tolist(),
            quat_i=[1, 0, 0, 0],
            quat_j=qj,
        )
        sim0, nematic0 = make_sim_with_nematic(device, snap0, epsilon, r_cut)

        forces = nematic0.forces
        if forces is None:
            return

        force_on_j = np.array(forces[1])

        # Numerical gradient: F_j = -dU/dr_j
        grad = np.zeros(3)
        for dim in range(3):
            for sign, coeff in [(+1, +1), (-1, -1)]:
                r_shift = r_base.copy()
                r_shift[dim] += sign * h
                snap_s = make_pair_snapshot(
                    device,
                    pos_i=[0, 0, 0],
                    pos_j=r_shift.tolist(),
                    quat_i=[1, 0, 0, 0],
                    quat_j=qj,
                )
                sim_s, nematic_s = make_sim_with_nematic(
                    device, snap_s, epsilon, r_cut
                )
                energies_s = nematic_s.energies
                if energies_s is not None:
                    grad[dim] += coeff * sum(energies_s)

        grad /= 2 * h
        force_numerical = -grad

        np.testing.assert_allclose(force_on_j, force_numerical, atol=1e-3)


class TestPolarMode:
    """Tests for multiplicity=1 (polar) mode."""

    def test_parallel_energy_polar(self, device):
        """Parallel orientations with multiplicity=1: U = -epsilon * g."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = -epsilon * 1.0 * g  # (n_i.n_j)^1 = 1

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut, multiplicity=1)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)

    def test_anti_parallel_energy_polar(self, device):
        """Anti-parallel with multiplicity=1: cos(pi) = -1, U = +epsilon * g > 0."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = -epsilon * (-1.0) * g  # = +epsilon * g > 0

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[0, 0, 0, 1],  # n_j = (-1,0,0)
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut, multiplicity=1)

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)
            assert total_e > 0, "Anti-parallel should be repulsive in polar mode"

    def test_polar_vs_nematic_symmetry_broken(self, device):
        """multiplicity=1 breaks nematic symmetry: E(parallel) != E(anti-parallel)."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5

        snap_par = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim_par, nem_par = make_sim_with_nematic(
            device, snap_par, epsilon, r_cut, multiplicity=1
        )

        snap_anti = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[0, 0, 0, 1],
        )
        sim_anti, nem_anti = make_sim_with_nematic(
            device, snap_anti, epsilon, r_cut, multiplicity=1
        )

        e_par = sum(nem_par.energies) if nem_par.energies is not None else None
        e_anti = sum(nem_anti.energies) if nem_anti.energies is not None else None
        if e_par is not None and e_anti is not None:
            assert e_par != pytest.approx(e_anti, abs=1e-6), \
                "Polar mode should distinguish parallel from anti-parallel"
            assert e_par < e_anti, "Parallel should be lower energy in polar mode"

    def test_torque_drives_polar_alignment(self, device):
        """In polar mode, torque should drive toward parallel (not anti-parallel)."""
        epsilon = 50.0
        r_cut = 5.0
        r = 1.5

        # Start 135 deg misaligned (closer to anti-parallel)
        theta = 3 * np.pi / 4
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],
        )

        sim = hoomd.Simulation(device=device)
        sim.create_state_from_snapshot(snap)

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        nematic = align_angle.DirectorPair(nlist=nlist, default_r_cut=r_cut)
        nematic.params[("A", "A")] = dict(epsilon=epsilon, multiplicity=1)

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.0)
        integrator = hoomd.md.Integrator(
            dt=0.001, methods=[langevin], forces=[nematic]
        )
        integrator.integrate_rotational_dof = True
        sim.operations.integrator = integrator
        sim.run(0)

        # Compute initial n_i . n_j (should be negative, ~-0.707)
        with sim.state.cpu_local_snapshot as ss:
            tags = np.array(ss.particles.tag[:])
            quats = np.array(ss.particles.orientation[:])
        order = np.argsort(tags)
        quats = quats[order]
        n0 = quat_rotate(quats[0], [1, 0, 0])
        n1 = quat_rotate(quats[1], [1, 0, 0])
        cos_before = np.dot(n0, n1)

        sim.run(200)

        with sim.state.cpu_local_snapshot as ss:
            tags = np.array(ss.particles.tag[:])
            quats = np.array(ss.particles.orientation[:])
        order = np.argsort(tags)
        quats = quats[order]
        n0 = quat_rotate(quats[0], [1, 0, 0])
        n1 = quat_rotate(quats[1], [1, 0, 0])
        cos_after = np.dot(n0, n1)

        assert cos_after > cos_before, (
            f"Polar mode: n_i.n_j should increase: {cos_before:.4f} -> {cos_after:.4f}"
        )

    def test_energy_formula_polar(self, device):
        """Verify energy matches the analytic formula for multiplicity=1."""
        epsilon = 7.0
        r_cut = 4.0

        theta = np.pi / 6
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)

        r_vec = np.array([1.0, 0.8, 0.3])
        r = np.linalg.norm(r_vec)

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=r_vec.tolist(),
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],
        )
        sim, nematic = make_sim_with_nematic(device, snap, epsilon, r_cut, multiplicity=1)

        n_i = np.array([1, 0, 0], dtype=float)
        n_j = quat_rotate([c, 0, 0, s], np.array([1, 0, 0], dtype=float))
        cos_nij = np.dot(n_i, n_j)
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected = -epsilon * cos_nij * g  # multiplicity=1: cos(alpha) = cos_nij

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected, rel=1e-5)

    def test_force_finite_difference_polar(self, device):
        """Compare force with numerical gradient of energy for multiplicity=1."""
        epsilon = 5.0
        r_cut = 3.5
        h = 1e-5

        r_base = np.array([1.2, 0.7, 0.4])
        theta = np.pi / 3
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        qj = [c, 0, s, 0]

        snap0 = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=r_base.tolist(),
            quat_i=[1, 0, 0, 0],
            quat_j=qj,
        )
        sim0, nematic0 = make_sim_with_nematic(
            device, snap0, epsilon, r_cut, multiplicity=1
        )

        forces = nematic0.forces
        if forces is None:
            return

        force_on_j = np.array(forces[1])

        grad = np.zeros(3)
        for dim in range(3):
            for sign, coeff in [(+1, +1), (-1, -1)]:
                r_shift = r_base.copy()
                r_shift[dim] += sign * h
                snap_s = make_pair_snapshot(
                    device,
                    pos_i=[0, 0, 0],
                    pos_j=r_shift.tolist(),
                    quat_i=[1, 0, 0, 0],
                    quat_j=qj,
                )
                sim_s, nematic_s = make_sim_with_nematic(
                    device, snap_s, epsilon, r_cut, multiplicity=1
                )
                energies_s = nematic_s.energies
                if energies_s is not None:
                    grad[dim] += coeff * sum(energies_s)

        grad /= 2 * h
        force_numerical = -grad

        np.testing.assert_allclose(force_on_j, force_numerical, atol=1e-3)


class TestDirectorPairPhase:
    """Tests for the phase parameter."""

    def test_phase_pi_repels_parallel(self, device):
        """m=1, phase=pi: at parallel (alpha=0), cos(0+pi)=-1, U = epsilon*g > 0."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = epsilon * g  # -epsilon * cos(pi) * g = epsilon * g

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[1, 0, 0, 0],
        )
        sim, nematic = make_sim_with_nematic(
            device, snap, epsilon, r_cut, multiplicity=1, phase=np.pi
        )

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)
            assert total_e > 0

    def test_phase_pi_attracts_anti_parallel(self, device):
        """m=1, phase=pi: at anti-parallel (alpha=pi), cos(2pi)=1, U = -epsilon*g."""
        epsilon = 5.0
        r_cut = 3.0
        r = 1.5
        x = 1.0 - r ** 2 / r_cut ** 2
        g = x ** 2
        expected_U = -epsilon * g  # -epsilon * cos(2pi) * g = -epsilon * g

        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[r, 0, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[0, 0, 0, 1],  # anti-parallel
        )
        sim, nematic = make_sim_with_nematic(
            device, snap, epsilon, r_cut, multiplicity=1, phase=np.pi
        )

        energies = nematic.energies
        if energies is not None:
            total_e = sum(energies)
            assert total_e == pytest.approx(expected_U, rel=1e-5)
            assert total_e < 0

    def test_newtons_third_law_with_phase(self, device):
        """Total force and torque are zero with phase != 0."""
        epsilon = 5.0
        r_cut = 3.0
        c = np.cos(np.pi / 8)
        s = np.sin(np.pi / 8)
        snap = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=[2.0, 0.5, 0],
            quat_i=[1, 0, 0, 0],
            quat_j=[c, 0, 0, s],
        )
        sim, nematic = make_sim_with_nematic(
            device, snap, epsilon, r_cut, multiplicity=2, phase=0.5
        )

        forces = nematic.forces
        if forces is not None:
            total_f = np.sum(forces, axis=0)
            np.testing.assert_allclose(total_f, [0, 0, 0], atol=1e-10)

        torques = nematic.torques
        if torques is not None:
            total_t = np.sum(torques, axis=0)
            np.testing.assert_allclose(total_t, [0, 0, 0], atol=1e-10)

    def test_force_finite_difference_with_phase(self, device):
        """Numerical gradient check with phase != 0."""
        epsilon = 5.0
        r_cut = 3.5
        h = 1e-5

        r_base = np.array([1.2, 0.7, 0.4])
        theta = np.pi / 3
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        qj = [c, 0, s, 0]

        snap0 = make_pair_snapshot(
            device,
            pos_i=[0, 0, 0],
            pos_j=r_base.tolist(),
            quat_i=[1, 0, 0, 0],
            quat_j=qj,
        )
        sim0, nematic0 = make_sim_with_nematic(
            device, snap0, epsilon, r_cut, multiplicity=2, phase=0.3
        )

        forces = nematic0.forces
        if forces is None:
            return

        force_on_j = np.array(forces[1])

        grad = np.zeros(3)
        for dim in range(3):
            for sign, coeff in [(+1, +1), (-1, -1)]:
                r_shift = r_base.copy()
                r_shift[dim] += sign * h
                snap_s = make_pair_snapshot(
                    device,
                    pos_i=[0, 0, 0],
                    pos_j=r_shift.tolist(),
                    quat_i=[1, 0, 0, 0],
                    quat_j=qj,
                )
                sim_s, nematic_s = make_sim_with_nematic(
                    device, snap_s, epsilon, r_cut, multiplicity=2, phase=0.3
                )
                energies_s = nematic_s.energies
                if energies_s is not None:
                    grad[dim] += coeff * sum(energies_s)

        grad /= 2 * h
        force_numerical = -grad

        np.testing.assert_allclose(force_on_j, force_numerical, atol=1e-3)