import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.split("\n")})

def code(src):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.split("\n")})

# Cell 1: Title
md("""
# External Patch Force \u2014 Filament Formation Demo

Simulates **500 linear "tripole" rods** in a flat 3D box (Lz = 5) that
self-assemble into **filaments** via outward-pointing patch interactions.

### Tripole unit (4 particles)

Each rod is modelled as **D\\_a \u2013 P\\_a \u2013 P\\_b \u2013 D\\_b**:

```
D_a \u2500\u2500\u2500(bond)\u2500\u2500\u2500 P_a \u2500\u2500\u2500(stiff bond)\u2500\u2500\u2500 P_b \u2500\u2500\u2500(bond)\u2500\u2500\u2500 D_b
 \u2191 outward patch         centre          outward patch \u2191
```

- **P** particles carry the attractive patch; their patch direction
  points *outward* along the rod axis (toward the paired D).
- **D** particles are phantom directors that define the patch direction.
- A stiff **P\u2013P** bond keeps the two halves co-linear.

Because each end of the rod has an outward-pointing patch, rods
bind *end-to-end*, forming **linear filaments**.

$$U_{ik} = f_i\\,f_k\\;\\epsilon\\!\\left(1 - r^2/r_c^2\\right)^2$$

with sigmoid angular envelopes
$f = \\bar\\sigma(\\omega\\,(\\hat p \\cdot \\hat r - \\cos\\alpha))$.

> **Why D\u2013P\u2013P\u2013D instead of P\u2013D\u2013P?**
> `ExternalPatch` computes the patch direction as
> $\\hat p = (\\text{director\\_pos} - \\text{particle\\_pos})/|\\cdots|$.
> Placing D at the centre would make both patches face *inward*,
> preventing inter-rod binding.  Putting D on the outside gives
> outward patches that drive end-to-end filament assembly.

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_tripoles | 500 | 2 000 particles total |
| Box | L \u00d7 L \u00d7 5 | Flat 3D slab |
| \u03b5 | 30 | ~30 kT \u2192 stable contacts |
| \u03b1 | 0.5 rad | \u2248 29\u00b0 half-angle |
| \u03c9 | 20 | Sharp sigmoid |
| r\\_cut (patch) | 1.5 | Patch attraction range |
| A (DPD, P\u2013P) | 25 | Soft repulsion |
| r\\_cut (DPD) | 1.0 | DPD cutoff |
| Bond k (P\u2013D) | 200 | Director tether |
| Bond k (P\u2013P) | 400 | Stiff centre bond |
| Bond r\u2080 | 0.5 | Both bond types |
| kT | 1.0 | Thermal energy |
""")

# Cell 2: Imports
code("""import sys
sys.path.insert(0, "/groups/goloborodko/user/anton.goloborodko/src/hoomd-blue/build/install_mixed/lib/python3.12/site-packages")

import numpy as np
import matplotlib.pyplot as plt
import hoomd
from hoomd import align_angle

print("HOOMD version:", hoomd.version.version)
""")

# Cell 3: Config markdown
md("""
## 1. Build the initial configuration

Place 500 tripole rods on a jittered grid in a flat 3D box (Lz = 5).
Each rod = 4 particles: D\\_a, P\\_a, P\\_b, D\\_b with random 3D orientation.
""")

# Cell 4: Build config
code("""
N_tripoles = 500
N = 4 * N_tripoles          # 2000 particles
bond_r0 = 0.5               # P-D director bond length
center_r0 = 0.5             # P-P centre bond length
Lz = 5.0                    # thin slab
grid_side = int(np.ceil(np.sqrt(N_tripoles)))  # 23
spacing = 1.3
L = grid_side * spacing + 4.0

device = hoomd.device.auto_select()
snap = hoomd.Snapshot(device.communicator)

if snap.communicator.rank == 0:
    snap.configuration.box = [L, L, Lz, 0, 0, 0]
    snap.particles.N = N
    snap.particles.types = ["P", "D"]
    snap.particles.mass[:] = 1.0

    rng = np.random.default_rng(42)
    positions = np.zeros((N, 3))
    typeids = np.zeros(N, dtype=int)

    for idx in range(N_tripoles):
        row = idx // grid_side
        col = idx % grid_side
        cx = (col - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
        cy = (row - grid_side / 2) * spacing + rng.uniform(-0.3, 0.3)
        cz = rng.uniform(-Lz / 2 + 0.5, Lz / 2 - 0.5)

        # Random 3D rod direction (uniform on sphere)
        cos_th = rng.uniform(-1, 1)
        sin_th = np.sqrt(1 - cos_th**2)
        phi = rng.uniform(0, 2 * np.pi)
        dx = sin_th * np.cos(phi)
        dy = sin_th * np.sin(phi)
        dz = cos_th

        i_Pa = 4 * idx       # P_a
        i_Da = 4 * idx + 1   # D_a
        i_Pb = 4 * idx + 2   # P_b
        i_Db = 4 * idx + 3   # D_b

        # Layout: D_a -- P_a -- centre -- P_b -- D_b
        half_c = center_r0 / 2
        positions[i_Pa] = [cx - half_c * dx, cy - half_c * dy, cz - half_c * dz]
        positions[i_Da] = [cx - (half_c + bond_r0) * dx,
                           cy - (half_c + bond_r0) * dy,
                           cz - (half_c + bond_r0) * dz]
        positions[i_Pb] = [cx + half_c * dx, cy + half_c * dy, cz + half_c * dz]
        positions[i_Db] = [cx + (half_c + bond_r0) * dx,
                           cy + (half_c + bond_r0) * dy,
                           cz + (half_c + bond_r0) * dz]

        typeids[i_Pa] = 0  # P
        typeids[i_Da] = 1  # D
        typeids[i_Pb] = 0  # P
        typeids[i_Db] = 1  # D

    snap.particles.position[:] = positions
    snap.particles.typeid[:] = typeids
    snap.particles.velocity[:] = rng.normal(0, 0.5, (N, 3))

    # Bonds: 3 per tripole
    snap.bonds.N = 3 * N_tripoles
    snap.bonds.types = ["PD", "PP"]
    snap.bonds.typeid[:] = 0
    for idx in range(N_tripoles):
        i_Pa = 4 * idx
        i_Da = 4 * idx + 1
        i_Pb = 4 * idx + 2
        i_Db = 4 * idx + 3
        b = 3 * idx
        snap.bonds.group[b]     = [i_Pa, i_Da]
        snap.bonds.typeid[b]    = 0               # PD
        snap.bonds.group[b + 1] = [i_Pa, i_Pb]
        snap.bonds.typeid[b + 1] = 1              # PP
        snap.bonds.group[b + 2] = [i_Pb, i_Db]
        snap.bonds.typeid[b + 2] = 0              # PD

print(f"N_tripoles = {N_tripoles}, N_particles = {N}, L = {L:.1f}, Lz = {Lz}")
""")

# Cell 5: Forces markdown
md("""
## 2. Set up forces and integrator
""")

# Cell 6: Forces
code("""sim = hoomd.Simulation(device=device, seed=42)
sim.create_state_from_snapshot(snap)

nlist_dpd = hoomd.md.nlist.Cell(buffer=0.4)
dpd = hoomd.md.pair.DPDConservative(nlist=nlist_dpd, default_r_cut=1.0)
dpd.params[("P", "P")] = dict(A=25.0)
dpd.params[("P", "D")] = dict(A=0.0)
dpd.params[("D", "D")] = dict(A=0.0)

bond_force = hoomd.md.bond.Harmonic()
bond_force.params["PD"] = dict(k=200.0, r0=bond_r0)
bond_force.params["PP"] = dict(k=400.0, r0=center_r0)

nlist_patch = hoomd.md.nlist.Cell(buffer=0.4)
patch = align_angle.ExternalPatch(nlist=nlist_patch, r_cut=1.5)
patch.epsilon = 30.0
patch.omega   = 20.0
patch.alpha   = 0.5
partners = []
for i in range(N_tripoles):
    partners.append((4 * i,     4 * i + 1))   # P_a -> D_a
    partners.append((4 * i + 2, 4 * i + 3))   # P_b -> D_b
patch.partners = partners

langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
langevin.gamma["P"] = 1.0
langevin.gamma["D"] = 1.0

integrator = hoomd.md.Integrator(dt=0.005, forces=[dpd, bond_force, patch],
                                  methods=[langevin])
sim.operations.integrator = integrator

thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermo)

print("Forces: DPDConservative (PP A=25), Harmonic bonds (PD k=200, PP k=400),")
print("        ExternalPatch (eps=30, alpha=0.5, omega=20, r_cut=1.5)")
""")

# Cell 7
md("""
## 3. Equilibrate and save snapshots

Run 500 000 steps (2 500 \u03c4) total.  Save snapshots at the start,
midpoint, and end.
""")

# Cell 8: Run
code("""
snapshots = {}

sim.run(0)
snapshots["t = 0"] = sim.state.get_snapshot()
ke = thermo.kinetic_energy / N
print(f"t = 0:     PE = {thermo.potential_energy:.1f},  KE/particle = {ke:.3f}")

sim.run(250_000)
snapshots["t = 1250 \\u03c4"] = sim.state.get_snapshot()
ke = thermo.kinetic_energy / N
print(f"t = 1250\\u03c4: PE = {thermo.potential_energy:.1f},  KE/particle = {ke:.3f}")

sim.run(250_000)
snapshots["t = 2500 \\u03c4"] = sim.state.get_snapshot()
ke = thermo.kinetic_energy / N
print(f"t = 2500\\u03c4: PE = {thermo.potential_energy:.1f},  KE/particle = {ke:.3f}")
""")

# Cell 9
md("""
## 4. Visualize filament formation

x\u2013y projection of the flat box.  Each panel shows:
- **Green** circles: P particles with an inter-rod contact
- **Red** circles: P particles without inter-rod contacts
- **Orange** dots + tethers: D (director) particles
- **Grey** lines: intra-rod P\u2013P bonds (rod axis)
- **Dark green** lines: inter-rod P\u2013P contacts (filament bonds)

A **zoomed-in** panel shows a random 10 \u00d7 10 region.
""")

# Cell 10: Visualization
code('''
def find_filaments(pos_P, box, N_tripoles, cutoff=1.3):
    """Find inter-tripole P-P contacts and connected-component filaments."""
    n = len(pos_P)
    Lx, Ly, Lz = box
    cutsq = cutoff * cutoff
    contacts = []
    tripole_adj = {}

    for i in range(n):
        ti = i // 2
        for j in range(i + 1, n):
            tj = j // 2
            if ti == tj:
                continue
            dx = pos_P[j, 0] - pos_P[i, 0]
            dy = pos_P[j, 1] - pos_P[i, 1]
            dz = pos_P[j, 2] - pos_P[i, 2]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dz -= Lz * np.round(dz / Lz)
            dsq = dx * dx + dy * dy + dz * dz
            if dsq < cutsq:
                contacts.append((i, j))
                tripole_adj.setdefault(ti, set()).add(tj)
                tripole_adj.setdefault(tj, set()).add(ti)

    visited = set()
    filaments = []
    for t in range(N_tripoles):
        if t in visited:
            continue
        component = []
        queue = [t]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nb in tripole_adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        filaments.append(component)
    return contacts, filaments


def plot_state(ax, snapshot, title, box, N_tripoles, cutoff=1.3, zoom=None):
    """Plot tripole rods with filament bonds (x-y projection)."""
    pos = np.array(snapshot.particles.position)
    tid = np.array(snapshot.particles.typeid)
    Lx, Ly, Lz = box

    pos_P = pos[tid == 0]
    pos_D = pos[tid == 1]

    contacts, filaments = find_filaments(pos_P, box, N_tripoles, cutoff)

    bonded_P = set()
    for i, j in contacts:
        bonded_P.add(i)
        bonded_P.add(j)

    if zoom is not None:
        s_P, s_D = 40, 10
        lw_inter, lw_intra, lw_tether = 3.5, 1.5, 1.0
    else:
        s_P, s_D = 6, 2
        lw_inter, lw_intra, lw_tether = 1.2, 0.4, 0.25

    # Intra-rod P_a-P_b bonds (grey)
    for k in range(N_tripoles):
        ia, ib = 2 * k, 2 * k + 1
        dx = pos_P[ib, 0] - pos_P[ia, 0]
        dy = pos_P[ib, 1] - pos_P[ia, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        ax.plot([pos_P[ia, 0], pos_P[ia, 0] + dx],
                [pos_P[ia, 1], pos_P[ia, 1] + dy],
                color="silver", lw=lw_intra, alpha=0.7, zorder=1)

    # P-D tether lines (orange)
    pdir = pos_D - pos_P
    pdir[:, 0] -= Lx * np.round(pdir[:, 0] / Lx)
    pdir[:, 1] -= Ly * np.round(pdir[:, 1] / Ly)
    pdir[:, 2] -= Lz * np.round(pdir[:, 2] / Lz)
    d_x = pos_P[:, 0] + pdir[:, 0]
    d_y = pos_P[:, 1] + pdir[:, 1]
    for k in range(len(pos_P)):
        ax.plot([pos_P[k, 0], d_x[k]], [pos_P[k, 1], d_y[k]],
                color="orange", lw=lw_tether, alpha=0.35, zorder=1)
    ax.scatter(d_x, d_y, c="orange", s=s_D, zorder=2,
               edgecolors="none", alpha=0.5)

    colors = ["green" if i in bonded_P else "red" for i in range(len(pos_P))]
    ax.scatter(pos_P[:, 0], pos_P[:, 1], c=colors, s=s_P, zorder=3,
               edgecolors="none")

    for i, j in contacts:
        dx = pos_P[j, 0] - pos_P[i, 0]
        dy = pos_P[j, 1] - pos_P[i, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        ax.plot([pos_P[i, 0], pos_P[i, 0] + dx],
                [pos_P[i, 1], pos_P[i, 1] + dy],
                color="darkgreen", lw=lw_inter, alpha=0.85, zorder=4,
                solid_capstyle="round")

    in_fil = sum(1 for f in filaments if len(f) >= 2 for _ in f)
    frac = in_fil / N_tripoles
    ax.set_title(f"{title}\\n{len(contacts)} bonds, {in_fil}/{N_tripoles} rods linked ({frac:.0%})",
                 fontsize=10)

    if zoom is not None:
        cx, cy, hw = zoom
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
    else:
        ax.set_xlim(-Lx / 2, Lx / 2)
        ax.set_ylim(-Ly / 2, Ly / 2)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)


box_dims = (L, L, Lz)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (label, snap_i) in zip(axes, snapshots.items()):
    plot_state(ax, snap_i, label, box_dims, N_tripoles)
fig.suptitle("Filament formation in a tripole gas (x-y projection)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

rng_zoom = np.random.default_rng(123)
half_box = L / 2
zoom_hw = 5.0
zx = rng_zoom.uniform(-half_box + zoom_hw, half_box - zoom_hw)
zy = rng_zoom.uniform(-half_box + zoom_hw, half_box - zoom_hw)

fig_z, ax_z = plt.subplots(figsize=(8, 8))
plot_state(ax_z, snapshots["t = 2500 \\u03c4"], "Zoomed in (final state)",
           box_dims, N_tripoles, zoom=(zx, zy, zoom_hw))
plt.tight_layout()
plt.show()
''')

# Cell 11
md("""
## 5. Filament length distribution & NN distances

- **Filament sizes** \u2014 connected components of tripoles linked by
  inter-rod P\u2013P contacts (cutoff 1.3).
- **NN distance** \u2014 nearest P\u2013P distance across different tripoles
  for each P particle.
""")

# Cell 12: Analysis
code('''
final_snap = snapshots["t = 2500 \\u03c4"]
pos = np.array(final_snap.particles.position)
tid = np.array(final_snap.particles.typeid)
pos_P = pos[tid == 0]

contacts, filaments = find_filaments(pos_P, (L, L, Lz), N_tripoles, cutoff=1.3)

fil_sizes = sorted([len(f) for f in filaments], reverse=True)
n_in_fil = sum(s for s in fil_sizes if s >= 2)

print(f"Total tripoles: {N_tripoles}")
print(f"Tripoles in filaments (size >= 2): {n_in_fil} ({n_in_fil / N_tripoles:.0%})")
print(f"Number of filaments (size >= 2): {sum(1 for s in fil_sizes if s >= 2)}")
print(f"Longest filament: {fil_sizes[0]} tripoles")
print(f"Mean filament size (>=2): {np.mean([s for s in fil_sizes if s >= 2]):.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

sizes_ge2 = [s for s in fil_sizes if s >= 2]
if sizes_ge2:
    ax1.hist(sizes_ge2, bins=range(2, max(sizes_ge2) + 2), color="seagreen",
             edgecolor="white", linewidth=0.5, align="left")
ax1.set_xlabel("Filament size (tripoles)", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_title("Filament size distribution", fontsize=13)

n_P = len(pos_P)
nn_dists = []
for i in range(n_P):
    ti = i // 2
    best_dsq = np.inf
    for j in range(n_P):
        if j == i or j // 2 == ti:
            continue
        dx = pos_P[j, 0] - pos_P[i, 0]
        dy = pos_P[j, 1] - pos_P[i, 1]
        dz = pos_P[j, 2] - pos_P[i, 2]
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        dz -= Lz * np.round(dz / Lz)
        dsq = dx * dx + dy * dy + dz * dz
        if dsq < best_dsq:
            best_dsq = dsq
    nn_dists.append(np.sqrt(best_dsq))
nn_dists = np.array(nn_dists)

ax2.hist(nn_dists, bins=60, range=(0, 4), color="steelblue",
         edgecolor="white", linewidth=0.3)
ax2.axvline(1.3, color="red", ls="--", lw=1, label="contact cutoff (1.3)")
ax2.set_xlabel("Nearest inter-rod P-P distance", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_title("NN distance distribution (final state)", fontsize=13)
ax2.legend()

plt.tight_layout()
plt.show()

bonded_frac = np.mean(nn_dists < 1.3)
print(f"\\nFraction of P particles with inter-rod NN < 1.3: {bonded_frac:.1%}")
''')

# Cell 13
md("""
## Summary

With outward-pointing patches (\u03b5 = 30, \u03b1 = 0.5 \u2248 29\u00b0) on each end of a
stiff 4-particle rod (D\u2013P\u2013P\u2013D), tripole units self-assemble end-to-end
into **linear filaments**.

The narrow patch half-angle prevents branching: each rod end can bind
at most one partner, so chains are strictly linear.

**To experiment:**
- Widen \u03b1 \u2192 1.0  to allow branching / sheet formation
- Lower \u03b5 \u2192 5    to see transient filaments that break and reform
- Increase N      for longer filaments at higher density
- Add angle forces to stiffen intra-rod P\u2013P\u2013D linearity
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("demo_external_patch_filament.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote demo_external_patch_filament.ipynb")
