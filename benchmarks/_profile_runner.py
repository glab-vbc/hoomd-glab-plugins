#!/usr/bin/env python
"""Lightweight profiling script for ExternalPatch GPU kernel.

Designed to be wrapped by nsys or ncu:

    nsys profile -o results/nsys_ep_1500 \\
        conda run -n main python _profile_runner.py --scenario externalpatch

    nsys profile -o results/nsys_ep_15000 \\
        conda run -n main python _profile_runner.py --scenario externalpatch --n-tripoles 5000

    /usr/local/cuda/bin/ncu --set full \\
        --kernel-name gpu_compute_external_patch_forces_kernel \\
        --launch-skip 50 --launch-count 5 \\
        -o results/ncu_ep_15000 \\
        conda run -n main python _profile_runner.py --scenario externalpatch --n-tripoles 5000 --steps 100

Usage:
    python _profile_runner.py [OPTIONS]

Options:
    --scenario {baseline,externalpatch,patchygaussian}  (default: externalpatch)
    --n-tripoles N        Number of P-D-P rods (default: 500 → 1500 particles)
    --warmup N            Warmup steps (default: 2000)
    --steps N             Profiled steps (default: 5000)
    --gpu-id N            GPU to use (default: 0)
"""
import sys
import argparse
import time

sys.path.insert(
    0,
    "/groups/goloborodko/user/anton.goloborodko/src/hoomd-blue"
    "/build/install_mixed/lib/python3.12/site-packages",
)

import numpy as np
import hoomd
from hoomd import align_angle


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Profile ExternalPatch kernel")
    p.add_argument(
        "--scenario",
        choices=["baseline", "baseline_lj15", "externalpatch", "patchygaussian"],
        default="externalpatch",
    )
    p.add_argument("--n-tripoles", type=int, default=500)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--shared-nlist", action="store_true",
                    help="Share one neighbor list between DPD and ExternalPatch")
    return p.parse_args()


# ── Snapshot builder ───────────────────────────────────────────────────
def build_snapshot(device, N_tripoles, Lz=5.0, bond_r0=0.5, spacing=1.3):
    N = 3 * N_tripoles
    grid_side = int(np.ceil(np.sqrt(N_tripoles)))
    L = grid_side * spacing + 4.0

    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, Lz, 0, 0, 0]
        snap.particles.N = N
        snap.particles.types = ["P", "D"]
        snap.particles.mass[:] = 1.0

        rng = np.random.default_rng(42)
        positions = np.zeros((N, 3))
        typeids = np.zeros(N, dtype=int)
        rod_dirs = np.zeros((N_tripoles, 3))

        for idx in range(N_tripoles):
            row, col = idx // grid_side, idx % grid_side
            cx = (col - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
            cy = (row - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
            cz = rng.uniform(-Lz / 2 + 0.5, Lz / 2 - 0.5)

            cos_th = rng.uniform(-1, 1)
            sin_th = np.sqrt(1 - cos_th**2)
            phi = rng.uniform(0, 2 * np.pi)
            dx, dy, dz = sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th
            rod_dirs[idx] = [dx, dy, dz]

            i_Pa, i_D, i_Pb = 3 * idx, 3 * idx + 1, 3 * idx + 2
            positions[i_D] = [cx, cy, cz]
            positions[i_Pa] = [cx - bond_r0 * dx, cy - bond_r0 * dy, cz - bond_r0 * dz]
            positions[i_Pb] = [cx + bond_r0 * dx, cy + bond_r0 * dy, cz + bond_r0 * dz]
            typeids[i_Pa] = 0; typeids[i_D] = 1; typeids[i_Pb] = 0

        # Wrap
        positions[:, 0] -= L * np.round(positions[:, 0] / L)
        positions[:, 1] -= L * np.round(positions[:, 1] / L)
        positions[:, 2] -= Lz * np.round(positions[:, 2] / Lz)

        snap.particles.position[:] = positions
        snap.particles.typeid[:] = typeids
        snap.particles.velocity[:] = rng.normal(0, 0.5, (N, 3))

        # Quaternion orientations (needed for PatchyGaussian)
        orientations = np.zeros((N, 4))
        orientations[:, 0] = 1.0
        for idx in range(N_tripoles):
            d = rod_dirs[idx]
            for i_P, outward in [(3 * idx, -d), (3 * idx + 2, d)]:
                ux = np.array([1.0, 0.0, 0.0])
                dot = np.clip(np.dot(ux, outward), -1.0, 1.0)
                if dot > 0.9999:
                    orientations[i_P] = [1, 0, 0, 0]
                elif dot < -0.9999:
                    orientations[i_P] = [0, 0, 0, 1]
                else:
                    axis = np.cross(ux, outward)
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(dot)
                    s, c = np.sin(angle / 2), np.cos(angle / 2)
                    orientations[i_P] = [c, axis[0] * s, axis[1] * s, axis[2] * s]
        snap.particles.orientation[:] = orientations

        moment = np.zeros((N, 3))
        for i in range(N):
            if typeids[i] == 0:
                moment[i] = [1, 1, 1]
        snap.particles.moment_inertia[:] = moment

        # Bonds
        snap.bonds.N = 2 * N_tripoles
        snap.bonds.types = ["PD"]
        snap.bonds.typeid[:] = 0
        for idx in range(N_tripoles):
            snap.bonds.group[2 * idx] = [3 * idx, 3 * idx + 1]
            snap.bonds.group[2 * idx + 1] = [3 * idx + 1, 3 * idx + 2]

        # Angles
        snap.angles.N = N_tripoles
        snap.angles.types = ["PDP"]
        snap.angles.typeid[:] = 0
        for idx in range(N_tripoles):
            snap.angles.group[idx] = [3 * idx, 3 * idx + 1, 3 * idx + 2]

    return snap


# ── Scenario setup ────────────────────────────────────────────────────
BOND_R0 = 0.5
DT = 0.005

def make_shared_forces(nlist=None):
    if nlist is None:
        nlist = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nlist, default_r_cut=1.0, kT=1.0)
    dpd.params[("P", "P")] = dict(A=40.0, gamma=1.0)
    dpd.params[("P", "D")] = dict(A=0.0, gamma=1.0)
    dpd.params[("D", "D")] = dict(A=0.0, gamma=1.0)
    bond = hoomd.md.bond.Harmonic()
    bond.params["PD"] = dict(k=200.0, r0=BOND_R0)
    angle = hoomd.md.angle.Harmonic()
    angle.params["PDP"] = dict(k=100.0, t0=np.pi)
    return nlist, dpd, bond, angle


def setup_baseline(device, snap):
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)
    _, dpd, bond, angle = make_shared_forces()
    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(
        dt=DT, forces=[dpd, bond, angle], methods=[method]
    )
    return sim


def setup_external_patch(device, snap, N_tripoles, shared_nlist=False):
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)

    nlist_patch = hoomd.md.nlist.Cell(buffer=0.4)
    nlist_for_dpd = nlist_patch if shared_nlist else None
    _, dpd, bond, angle = make_shared_forces(nlist=nlist_for_dpd)

    patch = align_angle.ExternalPatch(nlist=nlist_patch, r_cut=1.5)
    patch.epsilon = 20.0
    patch.width = 0.2
    partners = (
        [(3 * i, 3 * i + 1) for i in range(N_tripoles)]
        + [(3 * i + 2, 3 * i + 1) for i in range(N_tripoles)]
    )
    patch.partners = partners

    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(
        dt=DT, forces=[dpd, bond, angle, patch], methods=[method]
    )
    return sim


def setup_baseline_lj15(device, snap):
    """Baseline + weak attractive LJ at r_cut=1.5 on a shared nlist.

    This matches the neighbor-list radius of the EP scenario so that
    we can isolate the cost of the larger nlist from the EP kernel.
    """
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    _, dpd, bond, angle = make_shared_forces(nlist=nlist)

    lj = hoomd.md.pair.Gaussian(nlist=nlist, default_r_cut=1.5)
    lj.params[("P", "P")] = dict(epsilon=-0.1, sigma=0.5)
    lj.params[("P", "D")] = dict(epsilon=0.0, sigma=1.0)
    lj.params[("D", "D")] = dict(epsilon=0.0, sigma=1.0)

    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(
        dt=DT, forces=[dpd, bond, angle, lj], methods=[method]
    )
    return sim


def setup_patchy_gaussian(device, snap):
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)
    _, dpd, bond, angle = make_shared_forces()

    nlist_pg = hoomd.md.nlist.Cell(buffer=0.4)
    pg = hoomd.md.pair.aniso.PatchyGaussian(nlist=nlist_pg, default_r_cut=1.5)
    pg.params[("P", "P")] = dict(
        pair_params=dict(epsilon=-20.0, sigma=0.67),
        envelope_params=dict(alpha=0.45, omega=30.0),
    )
    pg.params[("P", "D")] = dict(
        pair_params=dict(epsilon=0.0, sigma=1.0),
        envelope_params=dict(alpha=0.45, omega=30.0),
    )
    pg.params[("D", "D")] = dict(
        pair_params=dict(epsilon=0.0, sigma=1.0),
        envelope_params=dict(alpha=0.45, omega=30.0),
    )
    pg.directors["P"] = [(1, 0, 0)]
    pg.directors["D"] = []

    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(
        dt=DT,
        forces=[dpd, bond, angle, pg],
        methods=[method],
        integrate_rotational_dof=True,
    )
    return sim


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    N_tripoles = args.n_tripoles
    N = 3 * N_tripoles

    print(f"HOOMD {hoomd.version.version}")
    print(f"Scenario: {args.scenario}")
    print(f"System: {N_tripoles} tripoles, {N} particles")
    print(f"Warmup: {args.warmup} steps, Profile: {args.steps} steps")
    print(f"GPU: {args.gpu_id}")

    device = hoomd.device.GPU(gpu_id=args.gpu_id)
    snap = build_snapshot(device, N_tripoles)
    print("Snapshot built.")

    if args.scenario == "baseline":
        sim = setup_baseline(device, snap)
    elif args.scenario == "baseline_lj15":
        sim = setup_baseline_lj15(device, snap)
    elif args.scenario == "externalpatch":
        sim = setup_external_patch(device, snap, N_tripoles,
                                   shared_nlist=args.shared_nlist)
    elif args.scenario == "patchygaussian":
        sim = setup_patchy_gaussian(device, snap)

    sim.run(0)  # attach
    print(f"Running {args.warmup} warmup steps...")
    sim.run(args.warmup)

    print(f"Running {args.steps} profiled steps...")
    t0 = time.perf_counter()
    sim.run(args.steps)
    elapsed = time.perf_counter() - t0
    tps = args.steps / elapsed
    print(f"Done: {elapsed:.3f} s, {tps:.0f} TPS")


if __name__ == "__main__":
    main()
