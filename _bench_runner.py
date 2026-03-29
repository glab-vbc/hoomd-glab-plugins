import sys
sys.path.insert(0, "/groups/goloborodko/user/anton.goloborodko/src/hoomd-blue/build/install_mixed/lib/python3.12/site-packages")

import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hoomd
from hoomd import align_angle

print("HOOMD version:", hoomd.version.version)

# ── Shared parameters ──────────────────────────────────────────────────
N_tripoles = 500
N = 3 * N_tripoles
bond_r0 = 0.5
Lz = 5.0
grid_side = int(np.ceil(np.sqrt(N_tripoles)))
spacing = 1.3
L = grid_side * spacing + 4.0

# ── Benchmark parameters ───────────────────────────────────────────────
WARMUP_STEPS = 1_000
BENCH_STEPS  = 10_000
N_REPEATS    = 3
DT           = 0.005

# ── ExternalPatch parameters ──────────────────────────────────────────
EP_EPSILON = 20.0
EP_WIDTH   = 0.2
EP_RCUT    = 1.5

# ── PatchyGaussian parameters (matched) ──────────────────────────────
PG_EPSILON = -20.0       # negative → attractive
PG_SIGMA   = 0.67        # Gaussian width
PG_ALPHA   = 0.45        # half-angle ≈ 26°
PG_OMEGA   = 30.0        # steepness
PG_RCUT    = 1.5

print(f"System: {N_tripoles} rods, {N} particles, box {L:.1f}×{L:.1f}×{Lz}")
print(f"Benchmark: {WARMUP_STEPS} warmup + {BENCH_STEPS} bench × {N_REPEATS} repeats")

def build_snapshot(device):
    """Build a P-D-P tripole snapshot with random rod orientations."""
    snap = hoomd.Snapshot(device.communicator)

    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, Lz, 0, 0, 0]
        snap.particles.N = N
        snap.particles.types = ["P", "D"]
        snap.particles.mass[:] = 1.0

        rng = np.random.default_rng(42)
        positions = np.zeros((N, 3))
        typeids = np.zeros(N, dtype=int)
        # Store rod directions for PatchyGaussian quaternion init
        rod_dirs = np.zeros((N_tripoles, 3))

        for idx in range(N_tripoles):
            row = idx // grid_side
            col = idx % grid_side
            cx = (col - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
            cy = (row - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
            cz = rng.uniform(-Lz / 2 + 0.5, Lz / 2 - 0.5)

            cos_th = rng.uniform(-1, 1)
            sin_th = np.sqrt(1 - cos_th**2)
            phi = rng.uniform(0, 2 * np.pi)
            dx = sin_th * np.cos(phi)
            dy = sin_th * np.sin(phi)
            dz = cos_th
            rod_dirs[idx] = [dx, dy, dz]

            i_Pa = 3 * idx
            i_D  = 3 * idx + 1
            i_Pb = 3 * idx + 2

            positions[i_D]  = [cx, cy, cz]
            positions[i_Pa] = [cx - bond_r0 * dx, cy - bond_r0 * dy, cz - bond_r0 * dz]
            positions[i_Pb] = [cx + bond_r0 * dx, cy + bond_r0 * dy, cz + bond_r0 * dz]

            typeids[i_Pa] = 0  # P
            typeids[i_D]  = 1  # D
            typeids[i_Pb] = 0  # P

        # Wrap positions
        positions[:, 0] -= L * np.round(positions[:, 0] / L)
        positions[:, 1] -= L * np.round(positions[:, 1] / L)
        positions[:, 2] -= Lz * np.round(positions[:, 2] / Lz)

        snap.particles.position[:] = positions
        snap.particles.typeid[:] = typeids
        snap.particles.velocity[:] = rng.normal(0, 0.5, (N, 3))

        # ── Quaternion orientations for PatchyGaussian ──
        # Rotate local +x → rod outward direction for each P particle
        orientations = np.zeros((N, 4))
        orientations[:, 0] = 1.0  # default: identity quaternion

        for idx in range(N_tripoles):
            d = rod_dirs[idx]
            # P_a outward direction: -d (pointing away from D toward P_a)
            # P_b outward direction: +d (pointing away from D toward P_b)
            for i_P, outward in [(3*idx, -d), (3*idx + 2, d)]:
                # Quaternion rotating +x → outward
                # Using axis-angle: axis = x × outward, angle = arccos(x · outward)
                ux = np.array([1.0, 0.0, 0.0])
                dot = np.clip(np.dot(ux, outward), -1.0, 1.0)
                if dot > 0.9999:
                    orientations[i_P] = [1, 0, 0, 0]
                elif dot < -0.9999:
                    orientations[i_P] = [0, 0, 0, 1]  # 180° around z
                else:
                    axis = np.cross(ux, outward)
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(dot)
                    s = np.sin(angle / 2)
                    c = np.cos(angle / 2)
                    orientations[i_P] = [c, axis[0]*s, axis[1]*s, axis[2]*s]

        snap.particles.orientation[:] = orientations

        # Moment of inertia for P particles (needed for rotational DOF)
        moment = np.zeros((N, 3))
        for i in range(N):
            if typeids[i] == 0:  # P
                moment[i] = [1, 1, 1]
        snap.particles.moment_inertia[:] = moment

        # ── Bonds ──
        snap.bonds.N = 2 * N_tripoles
        snap.bonds.types = ["PD"]
        snap.bonds.typeid[:] = 0
        for idx in range(N_tripoles):
            b = 2 * idx
            snap.bonds.group[b]     = [3*idx,     3*idx + 1]
            snap.bonds.group[b + 1] = [3*idx + 1, 3*idx + 2]

        # ── Angles ──
        snap.angles.N = N_tripoles
        snap.angles.types = ["PDP"]
        snap.angles.typeid[:] = 0
        for idx in range(N_tripoles):
            snap.angles.group[idx] = [3*idx, 3*idx + 1, 3*idx + 2]

    return snap, (rod_dirs if snap.communicator.rank == 0 else None)

device = hoomd.device.auto_select()
snap_template, rod_dirs = build_snapshot(device)
print(f"Snapshot ready: {N} particles, {2*N_tripoles} bonds, {N_tripoles} angles")
print(f"Device: {device}")

def make_shared_forces():
    """DPD + bonds + angles (common to all scenarios)."""
    nlist_dpd = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPD(nlist=nlist_dpd, default_r_cut=1.0, kT=1.0)
    dpd.params[("P", "P")] = dict(A=40.0, gamma=1.0)
    dpd.params[("P", "D")] = dict(A=0.0, gamma=1.0)
    dpd.params[("D", "D")] = dict(A=0.0, gamma=1.0)

    bond = hoomd.md.bond.Harmonic()
    bond.params["PD"] = dict(k=200.0, r0=bond_r0)

    angle = hoomd.md.angle.Harmonic()
    angle.params["PDP"] = dict(k=100.0, t0=np.pi)

    return dpd, bond, angle


def run_benchmark(sim, warmup, bench, label=""):
    """Warm up, then time `bench` steps.  Returns wall seconds."""
    sim.run(warmup)
    t0 = time.perf_counter()
    sim.run(bench)
    elapsed = time.perf_counter() - t0
    tps = bench / elapsed
    print(f"  {label}: {elapsed:.3f} s  ({tps:.0f} TPS)")
    return elapsed


def benchmark_scenario(name, setup_fn, n_repeats=N_REPEATS):
    """Run a scenario `n_repeats` times, return list of elapsed times."""
    print(f"\n{'='*60}")
    print(f"  Scenario: {name}")
    print(f"{'='*60}")
    times = []
    for rep in range(n_repeats):
        sim = setup_fn()
        sim.run(0)  # attach
        elapsed = run_benchmark(sim, WARMUP_STEPS, BENCH_STEPS,
                                label=f"rep {rep+1}/{n_repeats}")
        times.append(elapsed)
        del sim
    mean_t = np.mean(times)
    std_t  = np.std(times)
    tps_mean = BENCH_STEPS / mean_t
    print(f"  → mean {mean_t:.3f} ± {std_t:.3f} s  ({tps_mean:.0f} TPS)")
    return times

def setup_baseline():
    """Baseline: no patch force."""
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap_template)
    dpd, bond, angle = make_shared_forces()
    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(
        dt=DT,
        forces=[dpd, bond, angle],
        methods=[method],
    )
    sim.operations.integrator = integrator
    return sim


def setup_external_patch():
    """ExternalPatch: our 4-body Hermite smoothstep plugin."""
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap_template)
    dpd, bond, angle = make_shared_forces()

    nlist_patch = hoomd.md.nlist.Cell(buffer=0.4)
    patch = align_angle.ExternalPatch(nlist=nlist_patch, r_cut=EP_RCUT)
    patch.epsilon = EP_EPSILON
    patch.width   = EP_WIDTH
    partners = []
    for i in range(N_tripoles):
        partners.append((3*i,     3*i + 1))   # P_a → D
        partners.append((3*i + 2, 3*i + 1))   # P_b → D
    patch.partners = partners

    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(
        dt=DT,
        forces=[dpd, bond, angle, patch],
        methods=[method],
    )
    sim.operations.integrator = integrator
    return sim


def setup_patchy_gaussian():
    """PatchyGaussian: HOOMD built-in orientation-based patchy potential."""
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap_template)
    dpd, bond, angle = make_shared_forces()

    nlist_pg = hoomd.md.nlist.Cell(buffer=0.4)
    pg = hoomd.md.pair.aniso.PatchyGaussian(
        nlist=nlist_pg, default_r_cut=PG_RCUT,
    )
    pg.params[("P", "P")] = dict(
        pair_params=dict(epsilon=PG_EPSILON, sigma=PG_SIGMA),
        envelope_params=dict(alpha=PG_ALPHA, omega=PG_OMEGA),
    )
    # D particles have no patches — set zero interaction
    pg.params[("P", "D")] = dict(
        pair_params=dict(epsilon=0.0, sigma=1.0),
        envelope_params=dict(alpha=PG_ALPHA, omega=PG_OMEGA),
    )
    pg.params[("D", "D")] = dict(
        pair_params=dict(epsilon=0.0, sigma=1.0),
        envelope_params=dict(alpha=PG_ALPHA, omega=PG_OMEGA),
    )
    # Patch director along +x in local frame (quaternion rotates to world)
    pg.directors["P"] = [(1, 0, 0)]
    pg.directors["D"] = []

    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(
        dt=DT,
        forces=[dpd, bond, angle, pg],
        methods=[method],
        integrate_rotational_dof=True,
    )
    sim.operations.integrator = integrator
    return sim

print("Scenario setup functions defined.")

results = {}

results["Baseline"]       = benchmark_scenario("Baseline",       setup_baseline)
results["ExternalPatch"]  = benchmark_scenario("ExternalPatch",  setup_external_patch)
results["PatchyGaussian"] = benchmark_scenario("PatchyGaussian", setup_patchy_gaussian)

print("\n" + "="*60)
print("  All benchmarks complete")
print("="*60)

# ── Summary table ──────────────────────────────────────────────────────
print(f"{'Scenario':<20} {'Time (s)':>12} {'TPS':>10} {'Overhead':>10}")
print("-" * 55)

baseline_mean = np.mean(results["Baseline"])
for name, times in results.items():
    mean_t = np.mean(times)
    std_t  = np.std(times)
    tps    = BENCH_STEPS / mean_t
    overhead = (mean_t / baseline_mean - 1) * 100
    sign = "+" if overhead > 0 else ""
    print(f"{name:<20} {mean_t:>8.3f}±{std_t:.3f} {tps:>10.0f} {sign}{overhead:>8.1f}%")

# ── Bar chart: throughput (TPS) ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
means = [BENCH_STEPS / np.mean(t) for t in results.values()]
stds  = [BENCH_STEPS * np.std(t) / np.mean(t)**2 for t in results.values()]
colors = ["#4CAF50", "#2196F3", "#FF9800"]

bars1 = ax1.bar(names, means, yerr=stds, capsize=5, color=colors,
                edgecolor="white", linewidth=1.5)
ax1.set_ylabel("Throughput (time steps / sec)", fontsize=12)
ax1.set_title("Throughput comparison", fontsize=13)
ax1.tick_params(axis="x", rotation=15)
for bar, m in zip(bars1, means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stds)*0.3,
             f"{m:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ── Bar chart: overhead % over baseline ────────────────────────────────
overheads = [(np.mean(t) / baseline_mean - 1) * 100 for t in results.values()]
overhead_errs = [np.std(t) / baseline_mean * 100 for t in results.values()]

bars2 = ax2.bar(names, overheads, yerr=overhead_errs, capsize=5,
                color=colors, edgecolor="white", linewidth=1.5)
ax2.axhline(0, color="black", lw=0.8, ls="-")
ax2.set_ylabel("Overhead vs Baseline (%)", fontsize=12)
ax2.set_title("Computational overhead", fontsize=13)
ax2.tick_params(axis="x", rotation=15)
for bar, o in zip(bars2, overheads):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(overhead_errs)*0.3 if o >= 0 else bar.get_height() - max(overhead_errs)*0.5,
             f"{o:+.1f}%", ha="center", va="bottom" if o >= 0 else "top",
             fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig('bench_overhead.png', dpi=150, bbox_inches='tight')
print('Saved bench_overhead.png')
plt.close()

theta = np.linspace(0, np.pi, 500)
cos_theta = np.cos(theta)

# ── Hermite smoothstep: f(cos θ) with width w ─────────────────────────
w = EP_WIDTH
u_lo = 1 - w
t_h = np.clip((cos_theta - u_lo) / w, 0, 1)
f_hermite = 3 * t_h**2 - 2 * t_h**3

# ── Sigmoid: f(θ, α, ω)  (HOOMD Patchy convention) ───────────────────
alpha = PG_ALPHA
omega = PG_OMEGA
raw = 1.0 / (1.0 + np.exp(-omega * (cos_theta - np.cos(alpha))))
f_max = 1.0 / (1.0 + np.exp(-omega * (1 - np.cos(alpha))))
f_min = 1.0 / (1.0 + np.exp(-omega * (-1 - np.cos(alpha))))
f_sigmoid = (raw - f_min) / (f_max - f_min)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.degrees(theta), f_hermite, lw=2.5, label=f"Hermite (w={w})", color="#2196F3")
ax.plot(np.degrees(theta), f_sigmoid, lw=2.5, ls="--",
        label=f"Sigmoid (α={np.degrees(alpha):.0f}°, ω={omega})", color="#FF9800")
ax.axvline(np.degrees(np.arccos(1 - w)), color="#2196F3", lw=0.8, ls=":", alpha=0.6)
ax.axvline(np.degrees(alpha), color="#FF9800", lw=0.8, ls=":", alpha=0.6)
ax.set_xlabel("θ (degrees)", fontsize=12)
ax.set_ylabel("f(θ)  –  angular envelope", fontsize=12)
ax.set_title("Angular envelope comparison", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 90)
ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bench_angular_envelope.png', dpi=150, bbox_inches='tight')
print('Saved bench_angular_envelope.png')
plt.close()

# ── Radial profiles ───────────────────────────────────────────────────
r = np.linspace(0, 2.0, 500)
r_cut = EP_RCUT

# ExternalPatch radial
V_ep = np.where(r < r_cut, EP_EPSILON * (1 - (r / r_cut)**2)**2, 0.0)

# Gaussian radial (PatchyGaussian, ε < 0 → attractive)
V_pg = PG_EPSILON * np.exp(-r**2 / (2 * PG_SIGMA**2))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r, -V_ep, lw=2.5, label="ExternalPatch (negated for comparison)", color="#2196F3")
ax.plot(r, V_pg, lw=2.5, ls="--", label=f"PatchyGaussian (σ={PG_SIGMA})", color="#FF9800")
ax.axhline(0, color="black", lw=0.5)
ax.axvline(r_cut, color="grey", lw=0.8, ls=":", label=f"r_cut = {r_cut}")
ax.set_xlabel("r", fontsize=12)
ax.set_ylabel("V(r)  (attractive = negative)", fontsize=12)
ax.set_title("Radial pair potential comparison", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(0, 2.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bench_radial_potential.png', dpi=150, bbox_inches='tight')
print('Saved bench_radial_potential.png')
plt.close()
