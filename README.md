# hoomd-align-angle

> **Note:** This README was generated with the assistance of AI (GitHub Copilot)
> and may contain inaccuracies. Please verify the formulas against the source code.

A [HOOMD-blue](https://hoomd-blue.readthedocs.io/) plugin providing two
orientation-dependent forces for anisotropic particles:

1. **`DirectorAlign`** — aligns a particle's body axis to a direction defined by two
   guide particles (angle topology).
2. **`DirectorPair`** — an anisotropic pair potential that couples neighbouring
   particle orientations (nematic or polar symmetry).

Both forces run on **CPU and GPU** (HIP/CUDA).

---

## 1. DirectorAlign (angle force)

### Physics

For an angle group `(i, j, k)`:
- Particle `i` is the **oriented** particle whose body-frame x-axis `n̂ = rotate(q_i, x̂)`
  should align with the target direction.
- Particles `j` and `k` are **guide** particles that define the target direction
  `d̂ = (r_k − r_j) / |r_k − r_j|`.

The potential energy is:

$$U = \frac{K}{2}\bigl(1 - \cos(m\,\theta + \varphi_0)\bigr), \quad \theta = \arccos(\hat{n} \cdot \hat{d})$$

where $m$ is `multiplicity` (default 1) and $\varphi_0$ is `phase` (default 0).
With the defaults this reduces to $U = \frac{K}{2}(1 - \hat{n}\cdot\hat{d})$.

### Usage

```python
import hoomd
from hoomd import align_angle

align_force = align_angle.DirectorAlign()
align_force.params["align"] = dict(k=20.0)  # polar (default)

# Nematic (head-tail symmetric) alignment:
align_force.params["align"] = dict(k=20.0, multiplicity=2)

integrator = hoomd.md.Integrator(dt=0.005, methods=[...], forces=[align_force])
integrator.integrate_rotational_dof = True  # required!
```

---

## 2. DirectorPair (anisotropic pair potential)

### Physics

An orientation-dependent pair potential between particles within a cutoff
distance $r_c$:

$$U_{ij} = -\varepsilon \, \cos(m\,\alpha + \varphi_0) \; g(r)$$

with the smooth compact envelope

$$g(r) = \left(1 - \frac{r^2}{r_c^2}\right)^2$$

where $\alpha = \arccos(\hat{n}_i \cdot \hat{n}_j)$ is the angle between the
particle directors, $\hat{n} = \mathrm{rotate}(q, \hat{x})$ is each particle's
body-frame x-axis rotated into the lab frame, $m$ is `multiplicity` (default 1),
and $\varphi_0$ is `phase` (default 0).

| `multiplicity` | Symmetry | Minimum energy configuration |
|----------------|----------|------------------------------|
| 1 (default) | **Polar** | parallel only ($\alpha = 0$) |
| 2 | **Nematic** | parallel *or* anti-parallel ($\alpha = 0$ or $\pi$) |

The smooth compact envelope $g(r) = (1 - r^2/r_c^2)^2$ ensures both force and
energy vanish continuously at the cutoff.

### Usage

```python
from hoomd import align_angle

nlist = hoomd.md.nlist.Cell(buffer=0.4)

# Polar coupling (default, multiplicity=1):
polar = align_angle.DirectorPair(nlist=nlist, default_r_cut=1.5)
polar.params[("A", "A")] = dict(epsilon=4.0)

# Nematic coupling (multiplicity=2):
nematic = align_angle.DirectorPair(nlist=nlist, default_r_cut=1.5)
nematic.params[("A", "A")] = dict(epsilon=4.0, multiplicity=2)

integrator = hoomd.md.Integrator(dt=0.005, methods=[...], forces=[nematic])
integrator.integrate_rotational_dof = True  # required!
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epsilon` | float | — | Coupling strength $\varepsilon$ |
| `multiplicity` | int | 1 | Angular multiplicity $m$ (1 = polar, 2 = nematic) |
| `phase` | float | 0 | Phase offset $\varphi_0$ in radians |

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

38 tests total (15 DirectorAlign + 23 DirectorPair).

---

## Mathematical and Physical Details

This section provides the full derivations behind both forces, from physical
motivation to implementation-level detail.

### Motivation

Many soft-matter and biophysical systems involve **anisotropic particles** —
objects whose interactions depend not only on inter-particle distance but also on
orientation. Examples include:

- **Polymer segments** whose backbone tangent defines a preferred axis,
- **Liquid-crystal mesogens** that tend to align with their neighbours,
- **Elongated colloids** with direction-dependent attraction.

In coarse-grained models, each particle carries a **quaternion** $q$ that
describes its orientation. The **director** is a unit vector obtained by rotating
a reference body-frame axis into the lab frame:

$$\hat{n} = \mathrm{rotate}(q, \hat{x})$$

where $\hat{x} = (1,0,0)$ is the body-frame x-axis.

This plugin provides two complementary forces:

| Force | Topology | Purpose |
|-------|----------|---------|
| `DirectorAlign` | Angle $(i,j,k)$ | Steer particle $i$'s director toward an externally defined direction |
| `DirectorPair` | Pair $(i,j)$ | Couple neighbouring particle directors to each other |

Both potentials share the same `(multiplicity, phase)` parametrization of their
angular dependence, described next.

---

### The Generalised Angular Factor

Both forces use a potential of the form

$$f(\alpha) = \cos(m\,\alpha + \varphi_0)$$

where $\alpha$ is a geometric angle (between a director and a target direction,
or between two directors), $m \in \{1,2,3,\dots\}$ is the **multiplicity**, and
$\varphi_0$ is a **phase offset** in radians.

Useful special cases:

| $m$ | $\varphi_0$ | $f(\alpha)$ | Behaviour |
|-----|-------------|-------------|-----------|
| 1 | 0 | $\cos\alpha$ | **Polar**: minimum at $\alpha = 0$, maximum at $\alpha = \pi$ |
| 2 | 0 | $\cos 2\alpha$ | **Nematic**: minima at $\alpha = 0$ and $\pi$, maximum at $\pi/2$ |
| 1 | $\pi$ | $-\cos\alpha$ | **Anti-polar**: minimum at $\alpha = \pi$ |
| 2 | $\pi/2$ | $-\sin 2\alpha$ | Tilted equilibrium at $\alpha = \pi/4$ |

The derivative that enters torques and forces is

$$\frac{d f}{d \alpha} = -m\,\sin(m\,\alpha + \varphi_0).$$

---

### DirectorAlign — Full Derivation

#### Setup

An **angle group** $(i, j, k)$ defines two geometric objects:

1. The **director** of particle $i$: $\hat{n} = \mathrm{rotate}(q_i, \hat{x})$.
2. The **target direction** from guide particle $j$ to guide particle $k$:

$$\mathbf{d} = \mathbf{r}_k - \mathbf{r}_j, \qquad
\hat{d} = \frac{\mathbf{d}}{|\mathbf{d}|}.$$

(Minimum-image convention is applied to $\mathbf{d}$ for periodic boundaries.)

The angle between them is

$$\theta = \arccos(\hat{n} \cdot \hat{d}), \qquad \theta \in [0, \pi].$$

#### Potential energy

$$\boxed{U = \frac{K}{2}\bigl(1 - \cos(m\,\theta + \varphi_0)\bigr)}$$

This vanishes at the minimum ($m\,\theta + \varphi_0 = 0 \bmod 2\pi$)
and reaches $K$ at the maximum.

#### Torque on particle $i$

The potential depends on $i$'s orientation through $\hat{n}(\theta)$.
The orientational gradient gives a torque (in the lab frame):

$$\boldsymbol{\tau}_i
= -\frac{\partial U}{\partial \hat{n}} \times \hat{n}
= \frac{K}{2}\,\underbrace{\frac{m\,\sin(m\theta + \varphi_0)}{\sin\theta}}_{\displaystyle \equiv\, F_\theta}\;\hat{n} \times \hat{d}.$$

The factor $F_\theta$ generalises the simple $m{=}1$ result. When
$\sin\theta \to 0$ (parallel or anti-parallel), the cross product
$\hat{n} \times \hat{d}$ also vanishes, keeping the product finite;
numerically we set $F_\theta = 0$ when $\sin\theta < 10^{-8}$.

#### Forces on guide particles $j$ and $k$

The potential also depends on the direction $\hat{d}$, which is a function of
$\mathbf{r}_j$ and $\mathbf{r}_k$. Differentiating with respect to
$\mathbf{r}_j$ (with $\hat{d}$ depending on $\mathbf{r}_k - \mathbf{r}_j$):

$$\mathbf{F}_j
= -\frac{\partial U}{\partial \mathbf{r}_j}
= -\frac{K}{2}\,\frac{F_\theta}{|\mathbf{d}|}\,
  \bigl(\hat{n} - \cos\theta\;\hat{d}\bigr).$$

Newton's third law for the guide pair gives

$$\mathbf{F}_k = -\mathbf{F}_j.$$

**No translational force acts on particle $i$**, because $U$ depends on $i$
only through its orientation $q_i$.

#### Virial

The virial contribution is

$$W_{ab} = \frac{1}{3}\,F_j^{(a)}\,d^{(b)}$$

(the factor $\frac{1}{3}$ distributes the virial equally among the three
particles in the angle group, following HOOMD-blue convention).

---

### DirectorPair — Full Derivation

#### Setup

For a pair of anisotropic particles $i$ and $j$ separated by
$\mathbf{r} = \mathbf{r}_i - \mathbf{r}_j$ with $r = |\mathbf{r}| < r_c$,
define:

- **Directors**: $\hat{n}_i = \mathrm{rotate}(q_i, \hat{x})$, $\hat{n}_j = \mathrm{rotate}(q_j, \hat{x})$.
- **Inter-director angle**: $\alpha = \arccos(\hat{n}_i \cdot \hat{n}_j)$, $\alpha \in [0, \pi]$.
- **Radial envelope**: $g(r) = \bigl(1 - r^2/r_c^2\bigr)^2$.

The envelope $g(r)$ is a smooth, compactly-supported function that satisfies:

$$g(0) = 1, \qquad g(r_c) = 0, \qquad g'(r_c) = 0,$$

ensuring that both the energy and forces vanish continuously at the cutoff.
Explicitly, writing $x \equiv 1 - r^2/r_c^2$:

$$g = x^2, \qquad
\frac{dg}{dr} = -\frac{4\,x\,r}{r_c^2}.$$

#### Potential energy

$$\boxed{U = -\varepsilon\,\cos(m\,\alpha + \varphi_0)\;g(r)}$$

The factors separate into an angular part (coupling strength × angular
preference) and a radial part (distance modulation with smooth cutoff).

#### Radial force

The radial contribution to the force on particle $i$ comes from differentiating
$g(r)$ with respect to $\mathbf{r}$:

$$\mathbf{F}_i^{(\mathrm{radial})}
= -\frac{\partial U}{\partial \mathbf{r}}
= -\varepsilon\,\cos(m\alpha + \varphi_0)\,\frac{dg}{d\mathbf{r}}$$

$$= -\varepsilon\,\cos(m\alpha + \varphi_0) \cdot \left(-\frac{4x}{r_c^2}\right)\mathbf{r}$$

$$\boxed{\mathbf{F}_i^{(\mathrm{radial})} = \frac{4\varepsilon\,\cos(m\alpha+\varphi_0)\,x}{r_c^2}\;\mathbf{r}}$$

where $\mathbf{r} = \mathbf{r}_i - \mathbf{r}_j$.
When $\cos(m\alpha + \varphi_0) > 0$ (favourable angular alignment), the
force coefficient is negative (in the code, $f_{\mathrm{mag}} = -4\varepsilon\cos(m\alpha+\varphi_0)\,x/r_c^2 < 0$),
so the force points antiparallel to $\mathbf{r}$, i.e. **toward** particle $j$ — attractive, as
expected.

Newton's third law: $\mathbf{F}_j^{(\mathrm{radial})} = -\mathbf{F}_i^{(\mathrm{radial})}$.

#### Orientational torques

The angular part of $U$ depends on $\hat{n}_i$ and $\hat{n}_j$. Differentiating
with respect to the orientations:

$$\boldsymbol{\tau}_i
= -\frac{\partial U}{\partial \hat{n}_i} \times \hat{n}_i
= \varepsilon\,
  \underbrace{\frac{m\,\sin(m\alpha + \varphi_0)}{\sin\alpha}}_{\displaystyle T_\alpha}
  \;g(r)\;\hat{n}_i \times \hat{n}_j.$$

**Derivation sketch.** The chain rule gives

$$\frac{\partial U}{\partial \hat{n}_i}
= -\varepsilon\,g\,\frac{\partial}{\partial \hat{n}_i}\cos(m\alpha+\varphi_0)
= \varepsilon\,g\,m\sin(m\alpha+\varphi_0)\,\frac{\partial \alpha}{\partial \hat{n}_i}.$$

Since $\cos\alpha = \hat{n}_i \cdot \hat{n}_j$, we have
$\frac{\partial \alpha}{\partial \hat{n}_i} = -\hat{n}_j / \sin\alpha$ (projected
onto the plane perpendicular to $\hat{n}_i$). The cross product with $\hat{n}_i$
then yields $\hat{n}_i \times \hat{n}_j / \sin\alpha$.

The torque on $j$ follows by the equal-and-opposite principle for a pair that
depends on the angle between the directors:

$$\boldsymbol{\tau}_j = -\boldsymbol{\tau}_i.$$

#### Singularity at $\sin\alpha = 0$

When $\alpha \to 0$ or $\alpha \to \pi$ (parallel or anti-parallel directors),
$\sin\alpha \to 0$ and the ratio $\sin(m\alpha+\varphi_0)/\sin\alpha$ is
formally $0/0$.

Physically, the torque must vanish at these configurations because
$\hat{n}_i \times \hat{n}_j \to \mathbf{0}$. The product
$T_\alpha \cdot (\hat{n}_i \times \hat{n}_j)$ remains finite (by L'Hôpital:
$\lim_{\alpha\to 0} \sin(m\alpha)/\sin\alpha = m$, multiplied by
$|\hat{n}_i \times \hat{n}_j| = \sin\alpha \to 0$).

In the implementation, we set $T_\alpha = 0$ whenever $\sin\alpha < 10^{-8}$.

---

### Relationship to $\cos^p$ Formulations

Earlier versions of the code parameterised the nematic pair potential using an
integer power $p$:

$$U_{\mathrm{old}} = -\varepsilon\,(\hat{n}_i \cdot \hat{n}_j)^p\;g(r).$$

The `multiplicity` formulation generalises this. The two are related but **not
identical** for $p = m$:

| Old (`power`) | New (`multiplicity`) | Relationship |
|---------------|----------------------|--------------|
| $p = 1$ | $m = 1,\;\varphi_0 = 0$ | **Identical**: $\cos\alpha = \cos\alpha$ |
| $p = 2$ | $m = 2,\;\varphi_0 = 0$ | $\cos^2\alpha = \tfrac{1}{2}(1 + \cos 2\alpha)$, so $U_{\mathrm{new}} = 2\,U_{\mathrm{old}} + \varepsilon\,g$ |

For $m = 2$: the equilibria (minima, maxima, saddle points) are the same, but the
energy scale differs by a factor of 2. To recover identical dynamics, **halve
$\varepsilon$** compared to the old $p = 2$ parametrization.

The `multiplicity + phase` form is strictly more general because it can produce
angular dependencies (e.g. $\cos 3\alpha$, or $\sin 2\alpha$ via phase $= \pi/2$)
that no single integer power of $\cos\alpha$ can represent.

---

### Summary of All Equations

#### DirectorAlign

| Quantity | Expression |
|----------|------------|
| Energy | $U = \frac{K}{2}(1 - \cos(m\theta + \varphi_0))$ |
| Torque on $i$ | $\boldsymbol{\tau}_i = \frac{K}{2}\,\frac{m\sin(m\theta+\varphi_0)}{\sin\theta}\;\hat{n}\times\hat{d}$ |
| Force on $j$ | $\mathbf{F}_j = -\frac{K}{2}\,\frac{m\sin(m\theta+\varphi_0)}{\sin\theta\,|\mathbf{d}|}\;(\hat{n} - \cos\theta\;\hat{d})$ |
| Force on $k$ | $\mathbf{F}_k = -\mathbf{F}_j$ |
| Force on $i$ | $\mathbf{F}_i = \mathbf{0}$ |

where $\theta = \arccos(\hat{n}\cdot\hat{d})$, $\hat{n} = \mathrm{rotate}(q_i, \hat{x})$, $\hat{d} = (\mathbf{r}_k - \mathbf{r}_j)/|\mathbf{r}_k - \mathbf{r}_j|$.

#### DirectorPair

| Quantity | Expression |
|----------|------------|
| Energy | $U = -\varepsilon\,\cos(m\alpha+\varphi_0)\,g(r)$ |
| Envelope | $g(r) = (1 - r^2/r_c^2)^2$ |
| Radial force on $i$ | $\mathbf{F}_i = -\frac{4\varepsilon\,\cos(m\alpha+\varphi_0)\,(1-r^2/r_c^2)}{r_c^2}\;\mathbf{r}$ |
| Torque on $i$ | $\boldsymbol{\tau}_i = \varepsilon\,\frac{m\sin(m\alpha+\varphi_0)}{\sin\alpha}\,g(r)\;\hat{n}_i \times \hat{n}_j$ |
| Torque on $j$ | $\boldsymbol{\tau}_j = -\boldsymbol{\tau}_i$ |
| Force on $j$ | $\mathbf{F}_j = -\mathbf{F}_i$ |

where $\alpha = \arccos(\hat{n}_i \cdot \hat{n}_j)$, $\mathbf{r} = \mathbf{r}_i - \mathbf{r}_j$.

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
