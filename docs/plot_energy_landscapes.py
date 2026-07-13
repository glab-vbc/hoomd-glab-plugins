#!/usr/bin/env python3
# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.
"""Generate the energy-landscape figures embedded in README.md.

Each force's potential energy is plotted analytically (numpy only, no HOOMD)
against its natural variable. Run from anywhere:

    python docs/plot_energy_landscapes.py

Figures are written to docs/figures/*.png. The analytic expressions mirror the
shipped C++/CUDA evaluators (see src/SoftHarmonicTail.h, the per-class docstrings
in src/__init__.py, and README.md "Mathematical and Physical Details").
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Shared style -------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 2.0,
    }
)
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _finish(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)}")


# --- Shared soft/capped-harmonic core (mirrors src/SoftHarmonicTail.h) -----
def soft_U(x, k, xc, tail):
    s2 = (x / xc) ** 2
    if tail == "flat":
        return np.where(ax_abs(x) < xc, 0.5 * k * x**2 * (1 - s2 + s2**2 / 3), k * xc**2 / 6)
    return np.where(ax_abs(x) < xc, 0.5 * k * x**2, k * xc * ax_abs(x) - 0.5 * k * xc**2)


def soft_F(x, k, xc, tail):  # restoring force = -dU/dx
    if tail == "flat":
        return np.where(ax_abs(x) < xc, -k * x * (1 - (x / xc) ** 2) ** 2, 0.0)
    return np.where(ax_abs(x) < xc, -k * x, -k * xc * np.sign(x))


def ax_abs(x):
    return np.abs(x)


# ==========================================================================
# 1. DirectorAlign :  U(theta) = (K/2) (1 - cos(m*theta + phi0))
# ==========================================================================
def fig_directoralign():
    th = np.linspace(0, np.pi, 400)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (m, phi0, label), c in zip(
        [(1, 0.0, r"polar  $m{=}1,\ \varphi_0{=}0$"),
         (2, 0.0, r"nematic  $m{=}2$"),
         (1, np.pi, r"anti-polar  $m{=}1,\ \varphi_0{=}\pi$")],
        C,
    ):
        ax.plot(th, 0.5 * (1 - np.cos(m * th + phi0)), color=c, label=label)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    _finish(ax, "DirectorAlign: alignment energy",
            r"director angle  $\theta = \arccos(\hat n \cdot \hat d)$",
            r"$U / K$")
    _save(fig, "directoralign.png")


# ==========================================================================
# 2. SoftHarmonic (bond) :  U(r), F(r) for both tails vs plain harmonic
# ==========================================================================
def fig_softharmonic_bond():
    r0, xc, k = 1.0, 0.4, 1.0
    r = np.linspace(r0 - 2.0 * xc, r0 + 2.0 * xc, 500)
    x = r - r0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    a1.plot(r, 0.5 * k * x**2, "--", color="0.5", label="plain harmonic")
    a1.plot(r, soft_U(x, k, xc, "linear"), color=C[0], label='tail="linear" (Huber)')
    a1.plot(r, soft_U(x, k, xc, "flat"), color=C[1], label='tail="flat" (quartic)')
    a1.axvline(r0 + xc, color="0.7", lw=1, ls=":")
    a1.axvline(r0 - xc, color="0.7", lw=1, ls=":")
    _finish(a1, "SoftHarmonic bond: energy", r"bond length  $r$", r"$U / k$")

    a2.plot(r, -k * x, "--", color="0.5", label="plain harmonic")
    a2.plot(r, soft_F(x, k, xc, "linear"), color=C[0], label='"linear": force capped at $k x_c$')
    a2.plot(r, soft_F(x, k, xc, "flat"), color=C[1], label='"flat": force releases to 0')
    a2.axvline(r0 + xc, color="0.7", lw=1, ls=":")
    a2.axvline(r0 - xc, color="0.7", lw=1, ls=":")
    _finish(a2, "SoftHarmonic bond: restoring force", r"bond length  $r$", r"$F / k$")
    fig.suptitle(r"$r_0=1,\ x_c=0.4$   (dotted lines mark $r_0 \pm x_c$)", y=1.02)
    _save(fig, "softharmonic_bond.png")


# ==========================================================================
# 3. SoftHarmonicAngle :  U(theta) for both tails
# ==========================================================================
def fig_softharmonic_angle():
    t0, xc, k = np.pi / 2, 0.6, 1.0
    th = np.linspace(0, np.pi, 500)
    x = th - t0
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(th, 0.5 * k * x**2, "--", color="0.5", label="plain harmonic")
    ax.plot(th, soft_U(x, k, xc, "linear"), color=C[0], label='tail="linear"')
    ax.plot(th, soft_U(x, k, xc, "flat"), color=C[1], label='tail="flat"')
    ax.axvline(t0 + xc, color="0.7", lw=1, ls=":")
    ax.axvline(t0 - xc, color="0.7", lw=1, ls=":")
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(["0", r"$\theta_0=\pi/2$", r"$\pi$"])
    _finish(ax, r"SoftHarmonicAngle: energy  ($x_c=0.6$)", r"bond angle  $\theta$", r"$U / k$")
    _save(fig, "softharmonic_angle.png")


# ==========================================================================
# 4. DirectorPair :  U = -eps cos(m*alpha + phi0) g(r)
# ==========================================================================
def fig_directorpair():
    al = np.linspace(0, np.pi, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for (m, phi0, label), c in zip(
        [(1, 0.0, r"polar  $m{=}1$"),
         (2, 0.0, r"nematic  $m{=}2$"),
         (1, np.pi, r"anti-polar  $m{=}1,\ \varphi_0{=}\pi$")],
        C,
    ):
        a1.plot(al, -np.cos(m * al + phi0), color=c, label=label)  # g=1 (r=0)
    a1.set_xticks([0, np.pi / 2, np.pi])
    a1.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    _finish(a1, "DirectorPair: angular part (at $r=0$)",
            r"inter-director angle  $\alpha$", r"$U / \varepsilon$")

    rc = 1.5
    r = np.linspace(0, rc, 300)
    a2.plot(r, (1 - r**2 / rc**2) ** 2, color=C[3], label=r"$g(r)=(1-r^2/r_c^2)^2$")
    a2.axvline(rc, color="0.7", lw=1, ls=":")
    _finish(a2, "DirectorPair: radial envelope",
            r"separation  $r$  ($r_c=1.5$)", r"$g(r)$")
    _save(fig, "directorpair.png")


# ==========================================================================
# 5. SinSqDihedral :  U = (k/2)(1 + d cos(n*phi - phi0)) * sin^2 t1 * sin^2 t2
# ==========================================================================
def fig_sinsqdihedral():
    phi = np.linspace(-np.pi, np.pi, 500)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for (n, d, phi0, label), c in zip(
        [(1, -1, 0.0, r"$n{=}1,\ d{=}{-}1$ (cis)"),
         (1, 1, 0.0, r"$n{=}1,\ d{=}{+}1$ (trans)"),
         (3, -1, 0.0, r"$n{=}3,\ d{=}{-}1$")],
        C,
    ):
        a1.plot(phi, 0.5 * (1 + d * np.cos(n * phi - phi0)), color=c, label=label)  # sin^2=1
    a1.set_xticks([-np.pi, 0, np.pi])
    a1.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    _finish(a1, r"SinSqDihedral: torsion (at $\theta_1{=}\theta_2{=}90°$)",
            r"dihedral angle  $\phi$", r"$U / k$")

    th = np.linspace(0, np.pi, 300)
    a2.plot(th, np.sin(th) ** 2, color=C[4], label=r"$\sin^2\theta$")
    a2.set_xticks([0, np.pi / 2, np.pi])
    a2.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    _finish(a2, "Singularity-free envelope\n(one per central bond angle)",
            r"bond angle  $\theta$", r"$\sin^2\theta$")
    _save(fig, "sinsqdihedral.png")


# ==========================================================================
# 6. ExternalPatch :  U = -eps (1 - r^2/rc^2)^2 * f_i * f_k , Hermite envelope
# ==========================================================================
def _smoothstep(u, w):
    t = np.clip((u - (1 - w)) / w, 0.0, 1.0)
    return 3 * t**2 - 2 * t**3


def fig_externalpatch():
    rc = 3.0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    r = np.linspace(0, rc, 300)
    a1.plot(r, -(1 - r**2 / rc**2) ** 2, color=C[0], label=r"$-\,(1-r^2/r_c^2)^2$")
    a1.axvline(rc, color="0.7", lw=1, ls=":")
    _finish(a1, "ExternalPatch: radial well (fully aligned)",
            r"separation  $r$  ($r_c=3$)", r"$U / \varepsilon$")

    psi = np.linspace(0, np.pi / 2, 400)  # angle between patch dir and separation
    u = np.cos(psi)
    for w, c in zip([0.3, 0.5, 0.8], [C[1], C[2], C[3]]):
        a2.plot(np.degrees(psi), -_smoothstep(u, w), color=c, label=f"width $w={w}$")
    _finish(a2, "ExternalPatch: angular envelope (at $r=0$)",
            r"patch misalignment  $\psi$  (deg)", r"$U / \varepsilon$")
    _save(fig, "externalpatch.png")


if __name__ == "__main__":
    fig_directoralign()
    fig_softharmonic_bond()
    fig_softharmonic_angle()
    fig_directorpair()
    fig_sinsqdihedral()
    fig_externalpatch()
    print(f"\nAll figures written to {os.path.relpath(FIGDIR)}/")
