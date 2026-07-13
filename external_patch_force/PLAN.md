# Plan: External Patch Force for HOOMD-blue

## Summary

Implement a **four-body force** in HOOMD-blue as a custom `ForceCompute` in the
`hoomd-glab-plugins` plugin. Each designated particle *i* has a virtual "patch"
whose direction is the unit vector from *i* toward its partner *j* — no
quaternion degrees of freedom are needed. When two patched particles *i* and *k*
get close in space, they interact via a sigmoid angular envelope (identical to
HOOMD's `PatchEnvelope`) modulating a smooth radial potential
ε(1 − r²/rc²)². Forces are distributed across all four particles (*i, j, k, l*)
via the chain rule through the direction normalization.

The architecture follows the `AlignAngleForceCompute` / `PotentialValency`
pattern — a custom `ForceCompute` subclass that accepts a neighbor list, stores
a per-tag partner map in a `GPUArray`, and computes forces in a single
neighbor-list pass.

---

## Physics

### Energy

For a neighbor pair (*i*, *k*) with directors *j* = partner(*i*) and
*l* = partner(*k*):

    U_ik = f_i(p̂_i, r̂_ik) · f_k(p̂_k, r̂_ki) · ε (1 − r_ik²/rc²)²

where:
- p̂_i = (r_j − r_i) / |r_j − r_i|  (patch direction of *i*, set by partner *j*)
- p̂_k = (r_l − r_k) / |r_l − r_k|  (patch direction of *k*, set by partner *l*)
- f_i = σ̄(ω (p̂_i · r̂_ik − cos α))     sigmoid envelope, rescaled to [0,1]
- f_k = σ̄(ω (p̂_k · (−r̂_ik) − cos α))  note the sign flip so both patches "face" each other
- α = patch half-angle, ω = steepness

The sigmoid rescaling (to remove the non-zero baseline) and force-from-envelope
math are identical to `PatchEnvelope.h` in HOOMD.

### Forces on the four particles

Forces come from F_a = −∂U_ik/∂r_a for a ∈ {i, j, k, l}.

The energy depends on positions of all four particles:
- r_i, r_k: through the radial potential V(r) and the r̂_ik argument in f_i, f_k
- r_i, r_j: through p̂_i = normalize(r_j − r_i)
- r_k, r_l: through p̂_k = normalize(r_l − r_k)

Source decomposition:

1. **Radial force** (on i,k): −V'(r) r̂_ik modulated by the envelope f_i · f_k.
   Standard Newton's 3rd law pair between i and k.

2. **Envelope-position gradient** (on i,k): from ∂f_i/∂r̂_ik and ∂f_k/∂r̂_ik,
   identical to `PatchEnvelope::evaluate()` force output, but with the patch
   direction treated as a constant. This is also a pair force on i and k.

3. **Patch-direction gradient** (on i,j and on k,l): from the dependence of
   f_i on p̂_i through r_i and r_j. The key Jacobian is:

       ∂p̂/∂r_j = (I − p̂ p̂ᵀ) / d_ij

   where d_ij = |r_j − r_i|. This gives a force on j, and an equal-and-
   opposite force on i (since ∂p̂/∂r_i = −∂p̂/∂r_j). Similarly for k,l.

   The force on j from the patch channel is:

       F_j^patch = −V(r) · f_k · (∂f_i/∂p̂_i) · ∂p̂_i/∂r_j

   where ∂f_i/∂p̂_i = (df_i/du) · r̂_ik  (u = p̂_i · r̂_ik).

   Expand using the projection Jacobian:

       F_j^patch = −V(r) · f_k · (df_i/du) · [(r̂_ik − (p̂_i · r̂_ik) p̂_i)] / d_ij

   And F_i^patch = −F_j^patch (momentum conservation within the (i,j) pair).
   Analogous expressions hold for k and l.

### No torques

Since particles have no quaternion degrees of freedom, no torques are stored.
The "torque" on the virtual patch direction is fully resolved into translational
forces on the attractor (*i*) and its director partner (*j*).

### Momentum and energy conservation

- Total force on the quartet: F_i + F_j + F_k + F_l = 0 by construction.
  - Radial channel: F_i + F_k = 0 ✓
  - Envelope-position channel: F_i + F_k = 0 ✓
  - Patch channel for (i,j): F_i^patch + F_j^patch = 0 ✓
  - Patch channel for (k,l): F_k^patch + F_l^patch = 0 ✓
- Energy is a smooth function of all positions → forces are conservative →
  energy is conserved in NVE.

---

## Python API

```python
import hoomd
from hoomd import md
import align_angle

# Setup
nlist = md.nlist.Cell(buffer=0.4)

patch = align_angle.ExternalPatch(nlist=nlist, r_cut=3.0)

# Global interaction parameters
patch.params = dict(
    epsilon=5.0,      # attraction strength
    omega=20.0,       # sigmoid steepness
    alpha=0.5,        # patch half-angle (radians, ~29°)
)

# Partner assignments: (attractor_tag, director_tag)
# Particle attractor_tag gets a patch pointing toward director_tag.
# A particle can appear as attractor in one pair and director in another.
patch.partners = [
    (0, 1),   # particle 0's patch points toward particle 1
    (2, 3),   # particle 2's patch points toward particle 3
]

sim.operations.integrator.forces.append(patch)
```

### Python class structure

- `ExternalPatch` subclasses `hoomd.md.force.Force` directly (not `Pair` or
  `AnisotropicPair`) because this is a four-body interaction.
- `_cpp_class_name = "ExternalPatchForceCompute"`
- `_ext_module = _align_angle`
- Constructor accepts `nlist` and `r_cut`.
- `params` is a `ParameterDict` with keys: epsilon, omega, alpha, r_cut.
- `partners` is a Python list of (int, int) tuples, pushed to C++ via
  `self._cpp_obj.setPartners()`.
- `_attach_hook()` attaches the nlist (full storage mode), constructs the C++
  object, and pushes partners if set.

---

## Implementation Steps

### Step 1: Create `ExternalPatchForceCompute.h`

New file: `src/ExternalPatchForceCompute.h`

- Subclass `ForceCompute` (follow `AlignAngleForceCompute.h` pattern).
- Constructor: `(shared_ptr<SystemDefinition>, shared_ptr<NeighborList>)`.
- Members:
  - `GPUArray<int> m_partner_tag` — maps tag → partner tag (−1 = no partner).
    Resized to max particle tag count.
  - `shared_ptr<NeighborList> m_nlist`
  - Parameters struct: `epsilon`, `rcutsq`, `omega`, `cosalpha`
- Methods:
  - `setPartners(pybind11::list)` — populates `m_partner_tag`
  - `setParams(pybind11::dict)` / `getParams()` — parameter access
  - `getRequestedCommFlags()` — request `tag` in MPI ghosts
  - `computeForces(uint64_t timestep)` — the main computation

### Step 2: Implement `computeForces()`

In `ExternalPatchForceCompute.h` (or a `.cc` file).

Main loop (single pass, full neighbor list):

```
for each local particle i:
    tag_i = h_tag[i]
    partner_j_tag = m_partner_tag[tag_i]
    if partner_j_tag < 0: skip (no patch on this particle)
    idx_j = rtag[partner_j_tag]
    d_ij = minimum_image(pos[idx_j] - pos[i])
    dist_ij = |d_ij|
    if dist_ij < epsilon_guard: skip
    p_hat_i = d_ij / dist_ij                           // patch direction

    for each neighbor k of i in the neighbor list:
        tag_k = h_tag[k]
        partner_l_tag = m_partner_tag[tag_k]
        if partner_l_tag < 0: skip (k has no patch)
        idx_l = rtag[partner_l_tag]
        d_kl = minimum_image(pos[idx_l] - pos[k])
        dist_kl = |d_kl|
        if dist_kl < epsilon_guard: skip
        p_hat_k = d_kl / dist_kl                       // patch direction

        dr = minimum_image(pos[k] - pos[i])
        rsq = dot(dr, dr)
        if rsq >= rcutsq: skip

        --- compute f_i, f_k, V(r), derivatives ---
        --- compute forces on i, k, j, l ---
        --- accumulate into h_force arrays ---
```

Derivative computation reuses the math from `PatchEnvelope.h`:
- sigmoid values f_i, f_k
- derivatives df_i/du, df_k/du (u = cos angle between patch and r̂)
- radial V(r) = ε(1 − r²/rc²)² and V'(r)
- envelope force from ∂f/∂r̂ (quotient rule, same as PatchEnvelope)
- patch-direction force from ∂f/∂p̂ · ∂p̂/∂r (projection Jacobian)

Force accumulation:
- h_force[i] += F_i_radial + F_i_envelope + F_i_patch
- h_force[k] += F_k_radial + F_k_envelope + F_k_patch
- h_force[idx_j] += F_j_patch   (= −F_i_patch from i,j channel)
- h_force[idx_l] += F_l_patch   (= −F_k_patch from k,l channel)
- Energy: h_force[i].w += 0.5 * pair_eng; h_force[k].w += 0.5 * pair_eng
- Virial: set to 0 initially (NVT/NVE only)

Singularity guards: skip when dist_ij, dist_kl, or r_ik is below a small ε.

### Step 3: Create `ExternalPatchForceCompute.cc`

New file: `src/ExternalPatchForceCompute.cc`

- Constructor: store nlist, allocate m_partner_tag, register r_cut with nlist
- `setPartners()`: resize m_partner_tag if needed, fill from Python list
- `setParams()` / `getParams()`: convert dict ↔ struct
- `computeForces()`: can be here or in the header
- pybind11 export function `export_ExternalPatchForceCompute(module& m)`:
  ```cpp
  py::class_<ExternalPatchForceCompute, ForceCompute,
             shared_ptr<ExternalPatchForceCompute>>(m, "ExternalPatchForceCompute")
      .def(py::init<shared_ptr<SystemDefinition>, shared_ptr<NeighborList>>())
      .def("setPartners", &ExternalPatchForceCompute::setPartners)
      .def("setParams", &ExternalPatchForceCompute::setParams)
      .def("getParams", &ExternalPatchForceCompute::getParams);
  ```

### Step 4: Update build system

Modify `src/CMakeLists.txt`:
- Add `ExternalPatchForceCompute.cc` to `_${COMPONENT_NAME}_sources`
- Add `ExternalPatchForceCompute.h` to `_${COMPONENT_NAME}_headers`

Modify `src/module.cc`:
- Add `void export_ExternalPatchForceCompute(pybind11::module& m);` declaration
- Add `export_ExternalPatchForceCompute(m);` call in PYBIND11_MODULE

### Step 5: Create Python wrapper

Modify `src/__init__.py`:
- Add `ExternalPatch` class subclassing `hoomd.md.force.Force`
- Constructor accepts `nlist`, `r_cut`
- `params` via `ParameterDict(epsilon=float, omega=float, alpha=float, r_cut=float)`
- `partners` property — Python list, pushes to C++ via `setPartners()`
- `_attach_hook()`: attach nlist (full storage mode), construct C++ object,
  push partners

### Step 6: Verification notebook

New file: `demo_external_patch.ipynb`

#### 6a. Setup helper: create simulation with 4 particles

Create a reusable function `make_sim(positions)` that:
- Creates a 4-particle snapshot (type "A", box L=20)
- Positions particles at the given coordinates
- Partners: (0→1) and (2→3)
- Attaches `ExternalPatch` with epsilon=5.0, omega=20.0, alpha=0.5, r_cut=3.0
- Uses `hoomd.md.methods.ConstantVolume` with no thermostat (NVE)
- Returns `(sim, patch_force)` ready for `sim.run(0)` to trigger force evaluation

Test configuration: particles placed so that patches face each other:
- Particle 0 at (0, 0, 0), partner 1 at (0, 0, 2)  → patch points +z
- Particle 2 at (0, 0, 5), partner 3 at (0, 0, 7)  → patch points +z
- Particles 0 and 2 are within r_cut=3.0 of each other
- Patches face each other (0's patch points +z toward 2, 2's patch was +z
  but the envelope uses −r̂ so it correctly evaluates facing)

#### 6b. Finite-difference force test

For each of the 4 particles, for each of x,y,z:
1. Perturb position by +δ, run(0), read energy → U+
2. Perturb position by −δ, run(0), read energy → U−
3. Numerical force = −(U+ − U−) / (2δ)
4. Compare to analytical force from run(0) at the unperturbed position
5. Use δ = 1e-5, require relative error < 1e-4 (or absolute if force is small)

This tests all three force channels:
- Radial force (on i, k)
- Envelope-position gradient (on i, k)
- Patch-direction gradient (on i,j and k,l)

Note: each perturbation requires a fresh sim or reinitializing the snapshot,
since HOOMD caches neighbor lists. Alternatively, use `sim.state.set_snapshot()`
to reset positions between perturbations.

#### 6c. Momentum conservation test

At the unperturbed configuration:
1. Read forces on all 4 particles
2. Verify ΣF_x = ΣF_y = ΣF_z = 0 to machine precision (~1e-12)

#### 6d. NVE energy conservation test

1. Set up a ~20 particle system with several patch pairs
2. Give particles random initial velocities at kT=1
3. Run NVE for 10,000 steps with dt=0.001
4. Log kinetic + potential energy every 10 steps
5. Verify (E_max − E_min) / |E_mean| < 1e-4

This is the most stringent test — any sign error or missing chain-rule term
will cause energy drift.

#### 6e. Edge cases

- Particle without a partner: verify it contributes zero force
- All 4 particles at the same position: verify no crash (singularity guard)
- Particles exactly at r_cut: verify force and energy are zero
- Partners very close (d_ij ~ 0): verify no NaN/crash

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| No quaternions | Patch direction = r̂_ij. "Torque" manifests as non-central forces on i and j. Particles need no moment of inertia. |
| Custom ForceCompute (not AnisoPotentialPair) | Four-body nature prevents use of the two-body aniso evaluator template. |
| Partner storage by particle tag | Persistent tags are stable across domain decomposition, sorting, and migration. getRTag() gives O(1) index lookup. |
| Full neighbor list | Avoids Newton's 3rd law complications when accumulating forces on 4 particles. |
| Virial = 0 initially | Correct four-body virial is complex; start with NVT/NVE only. NPT support is a follow-up. |
| Global params (not per-type-pair) | The interaction is between patches, not particle types. Per-type-pair can be added later. |

## Reference Files

| File | Role |
|---|---|
| `src/AlignAngleForceCompute.h/.cc` | Closest pattern — custom ForceCompute with topology-driven torques |
| `hoomd/md/PatchEnvelope.h` | Sigmoid envelope math (f_i, f_k, derivatives) — reuse directly |
| `hoomd/md/PairModulator.h` | Shows force/torque decomposition for envelope × radial |
| `hoomd/ForceCompute.h` | Base class: m_force, m_torque arrays, getRequestedCommFlags |
| `hoomd/md/NeighborList.h` | Neighbor list access pattern (head_list, nlist, n_neigh) |
| `src/module.cc` | Plugin pybind11 module — add export call here |
| `src/CMakeLists.txt` | Build config — add source files here |
| `src/__init__.py` | Python wrappers — add ExternalPatch class here |
| `sq_density_plugin/src/PotentialValency.h` | Pattern for custom ForceCompute with neighbor list and per-particle data |

## MPI Considerations

- Ghost communication must include particle *tags* (for partner lookup) —
  request via `getRequestedCommFlags()` with `comm_flag::tag = 1`.
- Ghost layer width must be ≥ r_cut. The partner particles j, l may be far from
  i, k — they must also be accessible. If partners are bonded neighbors
  (distance ~ 1 bond length), the standard ghost width suffices. For distant
  partners, the user must ensure the ghost width is large enough, or partners
  must be within the ghost layer. A diagnostic warning should be emitted if
  a partner is not found in the local+ghost particle set.

## Progress Tracking

All implementation progress is tracked in `CHANGELOG.md` in this directory.

---

## Step 7: GPU Implementation (simple atomicAdd approach)

Port `computeForces()` to a HIP/CUDA kernel. One thread per local particle *i*.
Forces on partner *j* and partner *l* are accumulated via atomicAdd (the
standard HOOMD GPU pattern for multi-body forces — see `PotentialValencyGPU`).

### Performance profile

| Access | Cost per neighbor pair | Notes |
|--------|----------------------|-------|
| Particle i data | Coalesced read | Thread-local, one read |
| Partner j position | **Random-access** (~200–400 ns) | `pos[rtag[partner_tag[tag_i]]]` — 2 indirections |
| Neighbor k position | Strided coalesced read | From nlist, same as any pair force |
| Partner l position | **Random-access** (~200–400 ns) | Same 2-indirection pattern as j |
| Force on i | Thread-local register | Written once at end — free |
| Force on k | Can skip — k's own thread handles the reverse pair (full nlist) | Free |
| Force on j | **atomicAdd** (~5–10 ns, no contention) | Only 1 thread writes to each j |
| Force on l | **atomicAdd** (~5–10 ns, low contention) | Multiple threads may share same l |

The **dominant cost** is the two random-access position lookups (j and l), not
the atomics. Both are unavoidable given the 4-body structure.

### Files to create

1. **`src/ExternalPatchForceGPU.cuh`** — Kernel driver declaration
2. **`src/ExternalPatchForceGPU.cu`** — Kernel implementation:
   - One thread per local particle; early-exit if no partner
   - Outer: look up partner j, compute p̂_i
   - Inner (neighbor loop): for each patched k with partner l, compute all
     3 force channels (same math as CPU)
   - Force on i: register accumulation → single write
   - Force on k: skip (full nlist symmetry)
   - Force on j, l: atomicAdd (with idx < N guard for ghosts)
   - d_force/d_virial zero-initialized via hipMemset before launch
   - Use `myAtomicAdd()` wrapper (same as `PotentialValencyGPU.cuh`)
3. **`src/ExternalPatchForceComputeGPU.h`** — inherits ExternalPatchForceCompute,
   adds Autotuner, overrides computeForces()
4. **`src/ExternalPatchForceComputeGPU.cc`** — acquires device handles, calls
   kernel driver, pybind11 export

### Build system changes

- `src/CMakeLists.txt` — add GPU files in the `if (ENABLE_HIP)` block
- `src/module.cc` — add export in `#ifdef ENABLE_HIP` blocks
- `src/__init__.py` — no changes (already dispatches to GPU class)

### Verification

- Run `test_external_patch.py` with `hoomd.device.GPU()`
- All 5 tests must pass with same tolerances
- Add timing comparison: 1000 particles, 1000 steps, CPU vs GPU

---

## Step 8: GPU Optimization (if Step 7 is too slow)

If profiling shows Step 7 is bottlenecked by random-access partner lookups,
switch to a **precomputed patch-direction** design.

### Strategy: precompute p̂ in a separate kernel

**Pass 1 (lightweight kernel — O(N_patched), once per timestep):**
For each patched particle i, compute and store:
- `d_patch_dir[idx]` = `{p̂_i.x, p̂_i.y, p̂_i.z, 1/d_ij}` (ForceReal4)
- `d_partner_idx[idx]` = local index of partner j (unsigned int)

Particles without partners get `d_patch_dir = {0,0,0,0}`, `d_partner_idx = NOT_LOCAL`.

This kernel does one random-access read per particle (for partner j), fully
parallel, no atomics.

**Pass 2 (main force kernel):**
Same structure as Step 7, but the inner loop replaces:
- `pos[rtag[partner_tags[tag_k]]]` → `d_patch_dir[idx_k]` (coalesced read)
- `rtag[partner_tags[tag_k]]` → `d_partner_idx[idx_k]` (coalesced read)

No random-access reads remain in the inner loop.

### Performance comparison

| Operation | Step 7 (simple) | Step 8 (precomputed) |
|-----------|-----------------|---------------------|
| Partner j lookup | 1 random read/particle | 1 random read/particle (Pass 1) |
| Partner l lookup | **1 random read/neighbor** | **1 coalesced read/neighbor** |
| p̂_k computation | normalize per neighbor | 1 coalesced read from d_patch_dir |
| Extra memory | None | ~20 bytes/particle |
| Extra kernel launch | 0 | 1 (lightweight) |

The key win: **O(N × n_neighbors) random reads** → **O(N) random reads** +
**O(N × n_neighbors) coalesced reads**. Can be 10–50× less memory latency in
the inner loop.

### When to use Step 8

Profile Step 7 with `nsys` / `ncu`. Trigger if:
- L2 cache miss rate > 50% in force kernel
- Total force kernel time > 2× a comparable pair force
- Otherwise Step 7 is good enough — don't optimize

---

## Step 9: Bug Fixes — Repulsive Potential & Direction Convention

### Bug 1: The radial potential is repulsive, not attractive

The current radial potential is:

    V(r) = +ε (1 − r²/rc²)²

This is **always positive** — a soft repulsive bump (maximum at r=0, zero at
rc). The force pushes aligned-patch particles **apart**. For self-assembly the
potential must be attractive:

    V(r) = −ε (1 − r²/rc²)²

**Fix:** Negate V(r) and dV/dr in both CPU (`ExternalPatchForceCompute.cc`) and
GPU (`ExternalPatchForceGPU.cu`) implementations.

CPU (line ~365):
```cpp
// Before:
Scalar Vr = epsilon * x * x;
Scalar dVdr = Scalar(-4) * epsilon * r_mag * x / rcutsq;
// After:
Scalar Vr = -epsilon * x * x;
Scalar dVdr = Scalar(4) * epsilon * r_mag * x / rcutsq;
```

GPU (line ~252):
```cpp
// Before:
ForceReal Vr = epsilon * x * x;
ForceReal dVdr = ForceReal(-4.0) * epsilon * r_mag * x / rcutsq;
// After:
ForceReal Vr = -epsilon * x * x;
ForceReal dVdr = ForceReal(4.0) * epsilon * r_mag * x / rcutsq;
```

### Bug 2: Direction convention gives inward-pointing patches

The current direction convention is:

    p̂_i = normalize(r_partner − r_particle)

This makes the patch point FROM the particle TOWARD its partner.

**Problem for P–D–P:** With D at centre and partners `(P_a, D)` / `(P_b, D)`,
both P patches point inward toward D. For rod-end binding between P_b of rod A
and P_a of rod B:

    Rod A:  P_a ← D_A → P_b  ~~~~~~~~  P_a ← D_B → P_b  :Rod B
               patches point inward         patches point inward

    r̂ = normalize(r_{P_a_B} − r_{P_b_A}) = →
    u_i = dot(p̂_{P_b_A}, r̂) = dot(←, →) = −1  →  f ≈ 0  ✗
    u_k = −dot(p̂_{P_a_B}, r̂) = −dot(→, →) = −1  →  f ≈ 0  ✗

The angular envelope never activates — patches point away from each other.

**Fix:** Reverse the convention to:

    p̂_i = normalize(r_particle − r_partner)

Now the patch points FROM the partner TOWARD the particle, i.e. outward from
centre for P–D–P.

**Both demos work without any partner or geometry changes:**

P–D–P filament (D at centre, partners `(P_a, D)` / `(P_b, D)`):

    p̂_{P_b_A} = normalize(r_{P_b_A} − r_{D_A}) = →  (outward)
    p̂_{P_a_B} = normalize(r_{P_a_B} − r_{D_B}) = ←  (outward)
    u_i = dot(→, →) = +1  →  f ≈ 1  ✓
    u_k = −dot(←, →) = +1  →  f ≈ 1  ✓

Dimer (D on outside, partners `(P, D)`):

    D_A sits outside P_A (away from binding site)
    p̂_{P_A} = normalize(r_{P_A} − r_{D_A}) = toward binding site  ✓
    p̂_{P_B} = normalize(r_{P_B} − r_{D_B}) = toward binding site  ✓
    u_i = +1, u_k = +1  →  both f ≈ 1  ✓

**Code change** (CPU, lines ~253–258):
```cpp
// Before:
d_ij.x = pos_j.x - pos_i.x;  // partner − particle
d_ij.y = pos_j.y - pos_i.y;
d_ij.z = pos_j.z - pos_i.z;
// After:
d_ij.x = pos_i.x - pos_j.x;  // particle − partner (outward)
d_ij.y = pos_i.y - pos_j.y;
d_ij.z = pos_i.z - pos_j.z;
```

Same change for `d_kl` (neighbor's patch direction, lines ~336–341).
Same change in GPU kernel (`ExternalPatchForceGPU.cu`).

### Recommended fix strategy

1. Fix **Bug 1** in C++: negate V(r) and dV/dr (repulsive → attractive)
2. Fix **Bug 2** in C++: reverse patch direction convention (4 code sites:
   `d_ij` and `d_kl` in both CPU and GPU)
3. Both demos: **no changes needed** — partner assignments and geometry
   already work with the corrected convention
4. Update and re-run verification tests (Python reference energy is now
   negative; finite-diff baselines may shift sign)
5. Re-run both demo notebooks

### Verification after fixes

- Re-run `test_external_patch.py` — update Python reference energy (now
  negative) and force finite-diff baseline
- Verify NVE energy conservation still holds
- Re-run dimer demo: should show genuine patch-driven dimer formation
- Re-run filament demo: should show genuine filament formation driven by
  patch attraction (not DPD crowding artifacts)
- Sanity check: at t=0 (random init), linking fraction should be low;
  it should increase over time as patches find each other
