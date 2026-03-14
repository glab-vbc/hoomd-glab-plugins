## Plan: Orientation-Aligning Angle Force (C++ GPU plugin for HOOMD-blue)

### TL;DR

Implement a new C++ angle force (`AlignAngleForceCompute`) that reads the orientation quaternion of one particle in an angle triplet and the positions of the other two particles, then applies torques (on the oriented particle) and forces (on the guide particles) to align the particle's body-frame axis along the direction defined by the two guide particles. Uses HOOMD's angle topology `(i, j, k)` where particle `j` is the oriented particle and particles `i`, `k` define the target direction. The potential is $U = \frac{k}{2}(1 - \cos\theta)$ where $\theta$ is the angle between the oriented particle's local x-axis (rotated to world frame via its quaternion) and the direction $\hat{d} = \frac{\vec{r}_k - \vec{r}_i}{|\vec{r}_k - \vec{r}_i|}$. Full Newton's 3rd law: torques on `j`, reactive forces on `i` and `k`.

Two implementation options — described below as **Option A** (external plugin) or **Option B** (patched into the HOOMD-blue source tree). The architecture is identical; only the CMake integration differs.

### Physics

Given an angle group `(i, j, k)`:
- **Direction vector:** $\vec{d} = \vec{r}_k - \vec{r}_i$, $\hat{d} = \vec{d}/|\vec{d}|$
- **Particle axis:** $\hat{n} = q_j \cdot \hat{e}_x \cdot q_j^*$ where $\hat{e}_x = (1,0,0)$ is the body-frame reference axis and $q_j$ is the orientation quaternion of particle `j`
- **Angle:** $\cos\theta = \hat{n} \cdot \hat{d}$
- **Potential:** $U = \frac{k}{2}(1 - \cos\theta)$
- **Torque on j:** $\vec{\tau}_j = -\frac{\partial U}{\partial q_j}$ which evaluates to $\vec{\tau}_j = \frac{k}{2}\hat{n} \times \hat{d}$
- **Forces on i,k** (Newton's 3rd law from orientation coupling to positions): derived from $-\nabla_{\vec{r}_i} U$ and $-\nabla_{\vec{r}_k} U$, which arise because $\hat{d}$ depends on $\vec{r}_i$ and $\vec{r}_k$.

The body-frame reference axis could be configurable (defaulting to `(1,0,0)`), but starting with a hardcoded x-axis is simpler.

### Steps

**1. Create `AlignAngleForceCompute.h`** — CPU class header
Modeled on `HarmonicAngleForceCompute.h`. Extends `ForceCompute`. Stores `m_angle_data` (from `m_sysdef->getAngleData()`), per-type parameter `k` (spring constant). Overrides `computeForces()` and `getRequestedCommFlags()` (requesting `comm_flag::orientation` and `comm_flag::tag`).

**2. Create `AlignAngleForceCompute.cc`** — CPU `computeForces()` + pybind11 export
- Get `ArrayHandle` for `h_pos`, `h_orientation` (from `m_pdata->getOrientationArray()`), `h_rtag`, `h_force`, `h_torque`, `h_virial`
- Zero `m_force`, `m_torque`, `m_virial`
- Loop over all angle groups. For each `(i, j, k)`:
  - Compute $\vec{d} = \text{minImage}(\vec{r}_k - \vec{r}_i)$ and normalize
  - Extract quaternion $q_j$ from `h_orientation.data[idx_j]` via `quat<Scalar>(h_orientation.data[idx_j])`
  - Compute body axis $\hat{n} = \text{rotate}(q_j, \hat{e}_x)$ using `rotate()` from `VectorMath.h`
  - Compute $\cos\theta = \text{dot}(\hat{n}, \hat{d})$, energy, torque, and forces
  - Write torque on `j` to `h_torque.data[idx_j]`
  - Write forces on `i` and `k` to `h_force.data[idx_i]` and `h_force.data[idx_k]`
  - Write energy (split 1/3 to each particle) and virial
- Export via pybind11 with `setParams`/`getParams`

**3. Create `AlignAngleForceComputeGPU.h`** — GPU class header
Modeled on `HarmonicAngleForceComputeGPU.h`. Inherits `AlignAngleForceCompute`. Holds `Autotuner<1>` and `GPUArray<Scalar>` for params.

**4. Create `AlignAngleForceComputeGPU.cc`** — GPU `computeForces()` dispatcher
Gets device `ArrayHandle`s for positions, orientations, forces, torques, virials, angle GPU table data, and params. Launches the CUDA kernel via the pattern in `HarmonicAngleForceComputeGPU.cc`. Exports via pybind11.

**5. Create `AlignAngleForceGPU.cuh`** — CUDA kernel declaration
Declares `gpu_compute_align_angle_forces()` returning `hipError_t`. Signature includes `d_force`, `d_torque`, `d_virial`, `d_pos`, `d_orientation`, box, angle list data, and params.

**6. Create `AlignAngleForceGPU.cu`** — CUDA kernel
One thread per particle. Each thread loops over its angles (via the GPU angle table). For each angle:
- Determine if this thread is particle `i`, `j`, or `k` (via `apos_list`)
- Load positions of all three particles, load orientation of particle `j`
- Compute the same physics as the CPU version
- Accumulate force (for `i` or `k` roles) or torque (for `j` role) into thread-local accumulators
- Write out `d_force[idx]` and `d_torque[idx]` at end

**7. Register in CMake and module-md.cc** (if patching into HOOMD source)
- Add `.cc`, `.cu`, `.h`, `.cuh` files to `CMakeLists.txt`
- Add `export_AlignAngleForceCompute()` and GPU variant declarations + calls in `module-md.cc`

**OR** if building as external plugin:
- Create a standalone `CMakeLists.txt` using the [hoomd-component-template](https://github.com/glotzerlab/hoomd-component-template) pattern

**8. Create Python wrapper class**
Add `Align` (or `OrientationAlign`) class to `angle.py` (or a new module if external plugin). Inherits `Angle`. Sets `_cpp_class_name = "AlignAngleForceCompute"`. Exposes `params` TypeParameter with key `k` (spring constant). Optionally also `axis` (body-frame axis to align, default `(1,0,0)`).

**9. Write unit tests**
- Test that a single oriented particle with a known quaternion and two guide particles at known positions produces the expected torque magnitude and direction
- Test energy conservation in a short NVE run
- Test that forces on guide particles satisfy Newton's 3rd law ($\sum \vec{F} = 0$, $\sum \vec{\tau} = 0$ for the system)

### Verification

- **Unit tests**: Create a minimal system with 3 particles, one with non-trivial orientation. Verify computed torque direction and magnitude against analytic result.
- **Energy conservation**: Run short NVE with `integrate_rotational_dof=True`. Energy should be conserved (no drift).
- **Newton's 3rd law check**: Sum all forces and all torques — must be zero for isolated system.
- **Build**: `cmake --build build/` should compile without errors on both CPU and GPU paths.

### Decisions

- **Angle topology convention**: particle `j` (the central particle in the angle group) is the oriented particle; `i` and `k` are the guide particles whose positions define the target direction $\hat{d} = (\vec{r}_k - \vec{r}_i)/|\vec{r}_k - \vec{r}_i|$.
- **Body-frame axis**: defaults to local x-axis `(1,0,0)`. Can be made a per-type parameter later.
- **Potential**: $U = \frac{k}{2}(1 - \cos\theta)$ — this avoids the singularity of the harmonic-in-angle form at $\theta=0$ and $\theta=\pi$.
- **Implementation**: Directly in the HOOMD-blue source tree (Option B) since you already have the full source checked out. This avoids the complexity of an external CMake configuration.
