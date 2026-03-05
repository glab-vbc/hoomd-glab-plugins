# hoomd-align-angle

A [HOOMD-blue](https://hoomd-blue.readthedocs.io/) plugin providing two
orientation-dependent forces for anisotropic particles:

1. **`Align`** — aligns a particle's body axis to a direction defined by two
   guide particles (angle topology).
2. **`NematicPair`** — an anisotropic pair potential that couples neighbouring
   particle orientations (nematic or polar symmetry).

Both forces run on **CPU and GPU** (HIP/CUDA).

---

## 1. Align (angle force)

### Physics

For an angle group `(i, j, k)`:
- Particles `i` and `k` are **guide** particles that define a target direction
  `d̂ = (r_k − r_i) / |r_k − r_i|`.
- Particle `j` is the **oriented** particle whose body-frame x-axis `n̂ = rotate(q_j, x̂)`
  should align with `d̂`.

The potential energy is:

$$U = \frac{K}{2}(1 - \cos\theta), \quad \cos\theta = \hat{n} \cdot \hat{d}$$

Forces and torques:
- **Torque on j** (lab frame): $\tau_j = \frac{K}{2}\,\hat{n} \times \hat{d}$
- **Force on i**: $F_i = -\frac{K}{2|d|}\left(\hat{n} - \cos\theta\,\hat{d}\right)$
- **Force on k**: $F_k = -F_i$ (Newton's 3rd law)
- **No force on j** (potential couples only to j's orientation)

### Usage

```python
import hoomd
from hoomd import align_angle

align_force = align_angle.Align()
align_force.params["align"] = dict(k=20.0)

integrator = hoomd.md.Integrator(dt=0.005, methods=[...], forces=[align_force])
integrator.integrate_rotational_dof = True  # required!
```

---

## 2. NematicPair (anisotropic pair potential)

### Physics

An orientation-dependent pair potential between particles within a cutoff
distance $r_c$:

$$U_{ij} = -\varepsilon \, (\hat{n}_i \cdot \hat{n}_j)^p \left(1 - \frac{r_{ij}^2}{r_c^2}\right)^2$$

where $\hat{n} = \mathrm{rotate}(q, \hat{x})$ is each particle's body-frame
x-axis rotated into the lab frame.

The `power` parameter $p$ selects the symmetry:

| `power` | Symmetry | Minimum energy configuration |
|---------|----------|------------------------------|
| 2 (default) | **Nematic** | parallel *or* anti-parallel ($\hat{n}_i \cdot \hat{n}_j = \pm 1$) |
| 1 | **Polar** | parallel only ($\hat{n}_i \cdot \hat{n}_j = +1$) |

The smooth compact envelope $g(r) = (1 - r^2/r_c^2)^2$ ensures both force and
energy vanish continuously at the cutoff.

### Usage

```python
from hoomd import align_angle

nlist = hoomd.md.nlist.Cell(buffer=0.4)

# Nematic coupling (default power=2):
nematic = align_angle.NematicPair(nlist=nlist, default_r_cut=1.5)
nematic.params[("A", "A")] = dict(epsilon=4.0, power=2)

# Polar coupling (power=1):
polar = align_angle.NematicPair(nlist=nlist, default_r_cut=1.5)
polar.params[("A", "A")] = dict(epsilon=4.0, power=1)

integrator = hoomd.md.Integrator(dt=0.005, methods=[...], forces=[nematic])
integrator.integrate_rotational_dof = True  # required!
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `epsilon` | float | Coupling strength |
| `power` | int | 1 = polar (linear), 2 = nematic (squared). Default: 2 |

---

## Building

Requires HOOMD-blue ≥ 5.0.0 built with GPU support (HIP/CUDA).

```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$(python -c "import hoomd; print(hoomd.__path__[0] + '/..')")
cmake --build build -j$(nproc)
cmake --install build
```

## Tests

```bash
python -m pytest src/pytest/test_align_angle.py src/pytest/test_nematic_pair.py
```

26 tests total (7 align + 19 nematic, including 6 polar-mode tests).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
