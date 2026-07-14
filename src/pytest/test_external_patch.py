#!/usr/bin/env python3
"""Verification tests for ExternalPatchForceCompute.

Tests
-----
6a  Setup helper — build a 4-particle system, compare energy to Python reference
6b  Finite-diff — compare C++ analytical forces to numerical gradient
6c  Momentum conservation — verify ΣF = 0 across all four particles
6d  NVE energy conservation — drift < threshold over 10 000 steps
6e  Edge cases — unpaired particle, r = r_cut, single pair
"""

import sys
import numpy as np
import pytest
import hoomd
from hoomd import glab_forces


def _gpu_available():
    """True if HOOMD can see at least one GPU (for pytest parametrization)."""
    try:
        return len(hoomd.device.GPU.get_available_devices()) > 0
    except Exception:
        return False


@pytest.fixture(
    params=[False] + ([True] if _gpu_available() else []),
    ids=lambda g: "gpu" if g else "cpu",
)
def use_gpu(request):
    """Run each test on CPU, and also on GPU when one is available.

    Standalone use (``python test_external_patch.py [--gpu]``) is unaffected — the
    ``__main__`` runner passes ``use_gpu`` explicitly and ignores this fixture.
    """
    return request.param


# ═══════════════════════════════════════════════════════════════════════════
#  Python reference:  energy calculation (double-precision)
# ═══════════════════════════════════════════════════════════════════════════

def _smoothstep(u, width):
    """Cubic Hermite smoothstep envelope: 0 for u ≤ 1−width, 1 for u ≥ 1."""
    u_lo = 1.0 - width
    t = np.clip((u - u_lo) / width, 0.0, 1.0)
    return 3.0 * t * t - 2.0 * t * t * t


def compute_energy_py(positions, partners, epsilon, width, r_cut):
    """Total patch energy via brute-force ordered-pair loop.

    The loop visits every ordered pair (i, k) of patched particles;
    the pair is double-counted, so the result is divided by 2.
    """
    pos = np.asarray(positions, dtype=np.float64)
    rcutsq = r_cut ** 2

    partner_map = {int(a): int(d) for a, d in partners}
    patched = sorted(partner_map)

    total = 0.0
    for i in patched:
        j = partner_map[i]
        for k in patched:
            if k == i:
                continue
            l = partner_map[k]

            d_ij = pos[i] - pos[j]   # particle − partner (outward)
            norm_ij = np.linalg.norm(d_ij)
            if norm_ij < 1e-12:
                continue
            p_hat_i = d_ij / norm_ij

            d_kl = pos[k] - pos[l]   # particle − partner (outward)
            norm_kl = np.linalg.norm(d_kl)
            if norm_kl < 1e-12:
                continue
            p_hat_k = d_kl / norm_kl

            dr = pos[k] - pos[i]
            rsq = np.dot(dr, dr)
            if rsq >= rcutsq or rsq < 1e-24:
                continue
            r = np.sqrt(rsq)
            r_hat = dr / r

            # Radial potential  V(r) = −ε (1 − r²/rc²)²  (attractive)
            x = 1.0 - rsq / rcutsq
            Vr = -epsilon * x * x

            # Envelopes (cubic Hermite smoothstep)
            u_i = np.dot(p_hat_i, r_hat)
            fi = _smoothstep(u_i, width)

            u_k = np.dot(p_hat_k, -r_hat)
            fk = _smoothstep(u_k, width)

            total += fi * fk * Vr

    return total / 2.0          # ordered pairs → halve


# ═══════════════════════════════════════════════════════════════════════════
#  Helper:  create a HOOMD Simulation
# ═══════════════════════════════════════════════════════════════════════════

def make_sim(positions, partners, epsilon=5.0, width=0.5,
             r_cut=3.0, L=100.0, velocities=None, method="langevin",
             use_gpu=False):
    """Return  (simulation, patch_force)."""
    device = hoomd.device.GPU() if use_gpu else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=42)

    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        N = len(positions)
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = N
        snap.particles.types = ["A"]
        snap.particles.typeid[:] = [0] * N
        snap.particles.position[:] = positions
        snap.particles.mass[:] = [1.0] * N
        if velocities is not None:
            snap.particles.velocity[:] = velocities

    sim.create_state_from_snapshot(snap)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    patch = glab_forces.ExternalPatch(nlist=nlist, r_cut=r_cut)
    patch.epsilon = epsilon
    patch.width   = width
    patch.partners = partners

    integrator = hoomd.md.Integrator(dt=0.001, forces=[patch])

    if method == "langevin":
        m = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
        m.gamma["A"] = 1.0
        integrator.methods.append(m)
    elif method == "nve":
        m = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator.methods.append(m)

    sim.operations.integrator = integrator
    return sim, patch


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6a — Setup + energy sanity check
# ═══════════════════════════════════════════════════════════════════════════

def test_setup(use_gpu):
    """Build a 4-particle system, check HOOMD energy ≈ Python reference."""
    print("=" * 60)
    print("Test 6a: Setup and energy sanity check")
    print("=" * 60)

    # Geometry: partner behind each patched particle (away from the gap).
    # With p̂ = normalize(particle − partner), patches face the gap.
    #   j ─── i ═════ k ─── l
    positions = np.array([
        [ 0.0,  0.0,  0.0],    # 0  patched   (i)
        [-1.0, -0.1, -0.1],    # 1  partner   (j) — behind i, away from k
        [ 2.0,  0.2,  0.1],    # 2  patched   (k)
        [ 3.0,  0.3,  0.2],    # 3  partner   (l) — behind k, away from i
    ])
    partners = [(0, 1), (2, 3)]
    eps, width, r_cut = 5.0, 0.5, 3.0

    sim, patch = make_sim(positions, partners, eps, width, r_cut,
                           use_gpu=use_gpu)
    sim.run(0)

    energies = patch.energies
    forces   = patch.forces
    E_hoomd  = float(np.sum(energies))
    E_py     = compute_energy_py(positions, partners, eps, width, r_cut)

    print(f"  HOOMD per-particle energies : {energies}")
    print(f"  HOOMD total energy          : {E_hoomd:.10f}")
    print(f"  Python reference energy     : {E_py:.10f}")
    print(f"  Forces:\n{forces}")

    assert abs(E_hoomd) > 1e-6, f"Energy is trivially zero ({E_hoomd})"
    assert np.isclose(E_hoomd, E_py, rtol=1e-5), \
        f"Energy mismatch: HOOMD={E_hoomd}, Python={E_py}"
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6b — Finite-difference force validation
# ═══════════════════════════════════════════════════════════════════════════

def test_finite_diff(use_gpu):
    """Compare C++ analytical forces to central-difference gradient of the
    Python reference energy (double-precision throughout).
    """
    print("=" * 60)
    print("Test 6b: Finite-difference force validation")
    print("=" * 60)

    # Use moderate width so envelopes sit in the interesting middle range
    # for a wider variety of geometries.
    # Partners are behind their patched particles with off-axis offsets
    # so that u_i, u_k ≈ 0.6–0.9 (smoothstep slope region).
    #   j ─── i ═══════ k ─── l
    positions = np.array([
        [ 0.0,  0.0,  0.0],    # 0  patched   (i)
        [-1.0, -2.0, -0.5],    # 1  partner   (j) — behind i, off-axis
        [ 2.0,  0.5,  0.3],    # 2  patched   (k)
        [ 3.0,  1.5,  0.3],    # 3  partner   (l) — behind k, off-axis
    ])
    partners = [(0, 1), (2, 3)]
    eps, width, r_cut = 5.0, 0.5, 3.0

    # --- Analytical forces from C++ ---
    sim, patch = make_sim(positions, partners, eps, width, r_cut,
                           use_gpu=use_gpu)
    sim.run(0)
    analytical = np.array(patch.forces, dtype=np.float64)

    # --- Numerical gradient (Python double-precision energy) ---
    delta = 1e-5
    numerical = np.zeros_like(positions, dtype=np.float64)
    for i in range(len(positions)):
        for d in range(3):
            pos_p = positions.copy()
            pos_p[i, d] += delta
            E_p = compute_energy_py(pos_p, partners, eps, width, r_cut)

            pos_m = positions.copy()
            pos_m[i, d] -= delta
            E_m = compute_energy_py(pos_m, partners, eps, width, r_cut)

            numerical[i, d] = -(E_p - E_m) / (2.0 * delta)

    print("  Analytical (HOOMD C++)          Numerical (Python fin-diff)")
    max_rel = 0.0
    max_abs = 0.0
    # For GPU (single-precision forces), near-zero components carry float
    # noise (~1e-7) which gives huge relative error.  Skip them.
    abs_threshold = 1e-4 if use_gpu else 1e-8
    for i in range(len(positions)):
        a = analytical[i]
        n = numerical[i]
        print(f"  p{i}  ({a[0]:+10.6f} {a[1]:+10.6f} {a[2]:+10.6f})"
              f"  ({n[0]:+10.6f} {n[1]:+10.6f} {n[2]:+10.6f})")
        for dd in range(3):
            abs_err = abs(a[dd] - n[dd])
            max_abs = max(max_abs, abs_err)
            ref = max(abs(n[dd]), abs(a[dd]))
            if ref > abs_threshold:
                max_rel = max(max_rel, abs_err / ref)

    print(f"  Max absolute error : {max_abs:.6e}")
    print(f"  Max relative error : {max_rel:.6e}")
    assert max_rel < 1e-4, f"Finite-diff FAILED: max_rel = {max_rel}"
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6c — Momentum conservation  (ΣF = 0)
# ═══════════════════════════════════════════════════════════════════════════

def test_momentum(use_gpu):
    """Total force on all particles must vanish."""
    print("=" * 60)
    print("Test 6c: Momentum conservation")
    print("=" * 60)

    positions = np.array([
        [ 0.0,  0.0,  0.0],
        [-1.0, -2.0, -0.5],
        [ 2.0,  0.5,  0.3],
        [ 3.0,  1.5,  0.3],
    ])
    partners = [(0, 1), (2, 3)]

    sim, patch = make_sim(positions, partners, width=0.5,
                           use_gpu=use_gpu)
    sim.run(0)

    forces = np.array(patch.forces, dtype=np.float64)
    total  = np.sum(forces, axis=0)

    print(f"  Forces:\n{forces}")
    print(f"  ΣF = ({total[0]:+.6e}, {total[1]:+.6e}, {total[2]:+.6e})")
    print(f"  |ΣF| = {np.linalg.norm(total):.6e}")
    assert np.allclose(total, 0, atol=1e-6), \
        f"Momentum conservation FAILED: ΣF = {total}"
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6d — NVE energy conservation
# ═══════════════════════════════════════════════════════════════════════════

def test_nve(use_gpu):
    """Run NVE with 20 particles for 10 000 steps; check energy drift."""
    print("=" * 60)
    print("Test 6d: NVE energy conservation")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 20
    L = 10.0
    positions  = rng.uniform(-L / 4, L / 4, (N, 3))
    velocities = rng.normal(0, 0.1, (N, 3))

    n_patched = N // 2
    partners = [(i, i + n_patched) for i in range(n_patched)]

    # Use moderate width to avoid extremely stiff forces.
    sim, patch = make_sim(positions, partners, epsilon=5.0, width=0.5,
                          L=L, velocities=velocities,
                          method="nve", use_gpu=use_gpu)

    # Use smaller dt for NVE stability
    sim.operations.integrator.dt = 0.0005

    thermo = hoomd.md.compute.ThermodynamicQuantities(
        filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    n_steps = 10_000
    sample_period = 100
    energies = []

    for _ in range(n_steps // sample_period):
        sim.run(sample_period)
        energies.append(thermo.kinetic_energy + thermo.potential_energy)

    energies = np.array(energies)
    E0 = energies[0]
    drift = np.abs(energies - E0) / max(abs(E0), 1e-10)
    max_drift = np.max(drift)

    print(f"  E_0     = {E0:.8f}")
    print(f"  E_final = {energies[-1]:.8f}")
    print(f"  Max relative drift : {max_drift:.6e}")
    assert max_drift < 1e-3, f"NVE FAILED: max drift = {max_drift}"
    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Test 6e — Edge cases
# ═══════════════════════════════════════════════════════════════════════════

def test_edge_cases(use_gpu):
    """No-partner particle, pair at r_cut, single pair (no self-energy)."""
    print("=" * 60)
    print("Test 6e: Edge cases")
    print("=" * 60)

    # --- Case 1: unpaired particle ---
    print("  Case 1: unpaired particle near paired ones …")
    positions = np.array([
        [ 0.0,  0.0,  0.0],        # 0  patched
        [-1.0, -0.1, -0.1],        # 1  partner of 0
        [ 2.0,  0.2,  0.1],        # 2  patched
        [ 3.0,  0.3,  0.2],        # 3  partner of 2
        [ 0.5,  0.5,  0.0],        # 4  NO partner, nearby
    ])
    sim, patch = make_sim(positions, [(0, 1), (2, 3)], use_gpu=use_gpu)
    sim.run(0)
    f4 = np.array(patch.forces)[4]
    e4 = np.array(patch.energies)[4]
    assert np.allclose(f4, 0, atol=1e-12), f"Unpaired has force: {f4}"
    assert abs(e4) < 1e-12, f"Unpaired has energy: {e4}"
    print("    Force/energy on unpaired particle = 0  ✓")

    # --- Case 2: pair at r = r_cut → energy must vanish ---
    print("  Case 2: pair at r = r_cut …")
    r_cut = 3.0
    positions2 = np.array([
        [0.0,   0.0, 0.0],
        [0.0,   0.0, 2.0],
        [r_cut, 0.0, 0.0],
        [r_cut, 0.0, 2.0],
    ])
    sim2, patch2 = make_sim(positions2, [(0, 1), (2, 3)], r_cut=r_cut,
                             use_gpu=use_gpu)
    sim2.run(0)
    E_cut = float(np.sum(patch2.energies))
    # At r = r_cut, V(r) = 0, so the energy should vanish.
    # Due to to the nlist buffer, the pair might barely be excluded,
    # but the radial envelope (1 − r²/rc²)² is exactly 0 at r = rc.
    assert abs(E_cut) < 1e-10, f"Energy at r_cut should vanish, got {E_cut}"
    print(f"    Energy at r = r_cut: {E_cut:.2e}  ✓")

    # --- Case 3: single pair (only 2 particles) → no self-energy ---
    print("  Case 3: single pair (no self-energy) …")
    positions3 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    sim3, patch3 = make_sim(positions3, [(0, 1)], use_gpu=use_gpu)
    sim3.run(0)
    E_single = float(np.sum(patch3.energies))
    assert abs(E_single) < 1e-12, f"Single pair has energy: {E_single}"
    print(f"    Energy with one pair: {E_single:.2e}  ✓")

    print("  PASSED\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    use_gpu = "--gpu" in sys.argv
    if use_gpu:
        print(">>> Running tests on GPU <<<\n")
    passed, failed = [], []
    for name, fn in [
        ("6a-setup",    lambda: test_setup(use_gpu)),
        ("6b-findiff",  lambda: test_finite_diff(use_gpu)),
        ("6c-momentum", lambda: test_momentum(use_gpu)),
        ("6d-nve",      lambda: test_nve(use_gpu)),
        ("6e-edges",    lambda: test_edge_cases(use_gpu)),
    ]:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            failed.append((name, e))
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}\n")

    print("=" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    for p in passed:
        print(f"  ✓ {p}")
    for name, e in failed:
        print(f"  ✗ {name}: {e}")
    print("=" * 60)
    sys.exit(1 if failed else 0)
