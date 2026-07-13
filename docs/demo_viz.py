"""Shared *visualization + analysis* helpers for the align_angle demo notebooks.

This module deliberately contains **no simulation code**.  Every function takes
plain numpy arrays (positions, orientations, ...) that a notebook has already
collected, so the *simulation half* of each demo stays fully self-contained and
can be lifted out and reused without importing this module.  Import it only in
the analysis / visualization section, after the run:

    import demo_viz as v
    dirs   = v.director_from_quat(quats)     # (M, 3) body-x axes
    S, n   = v.nematic_order(dirs)           # scalar order parameter + axis
    v.render3d(pos, directors=dirs, connect=True)

Depends only on numpy + matplotlib (no hoomd, no fresnel), so it never ends up
on a simulation's critical path.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


def director_from_quat(quat):
    """Body-frame x-axis of each particle rotated into the lab frame.

    ``quat`` is a ``(..., 4)`` array of unit quaternions ``(s, x, y, z)``.
    Returns a ``(..., 3)`` array of unit directors ``n = rotate(q, x_hat)`` --
    the same quantity the ``DirectorAlign`` / ``DirectorPair`` forces use.
    """
    q = np.asarray(quat, dtype=float)
    s, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # First column of the quaternion rotation matrix (i.e. rotate x_hat=(1,0,0)):
    nx = 1.0 - 2.0 * (y * y + z * z)
    ny = 2.0 * (x * y + s * z)
    nz = 2.0 * (x * z - s * y)
    return np.stack([nx, ny, nz], axis=-1)


def unwrap(pos, image, box):
    """Absolute (unwrapped) coordinates from periodic-wrapped ``pos`` + ``image`` flags.

    ``box`` may be a full HOOMD box list ``[Lx, Ly, Lz, xy, xz, yz]`` or just the
    three lengths.  Needed before computing bonded quantities (tangents, bond
    lengths) or rendering a molecule that has drifted across the periodic box.
    """
    return np.asarray(pos, float) + np.asarray(image) * np.asarray(box, float)[:3]


def nematic_order(directors):
    """Nematic scalar order parameter ``S`` and director from orientation vectors.

    ``directors`` is an ``(M, 3)`` array of (not necessarily normalized) vectors,
    e.g. the output of :func:`director_from_quat`.  Returns ``(S, n)`` where
    ``S`` in ``[0, 1]`` is the largest eigenvalue of the nematic Q-tensor
    ``Q = <(3 nn - I) / 2>`` (``0`` isotropic, ``1`` perfectly aligned) and ``n``
    is the corresponding principal axis.
    """
    d = np.asarray(directors, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    Q = 1.5 * np.einsum("mi,mj->ij", d, d) / len(d) - 0.5 * np.eye(3)
    evals, evecs = np.linalg.eigh(Q)
    return float(evals[-1]), evecs[:, -1]


def _equal_aspect(ax, pos):
    """Give a 3d axis a cubic (equal-aspect) view around the data."""
    ctr = 0.5 * (pos.max(0) + pos.min(0))
    rad = 0.5 * float((pos.max(0) - pos.min(0)).max()) or 1.0
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1))


def render3d(pos, directors=None, color=None, connect=False, every=1,
             cmap="viridis", clim=None, clabel=None, colorbar=True, arrow_len=2.0,
             arrow_color="crimson", title=None, ax=None):
    """Static 3d render of a configuration (matplotlib; fresnel-free).

    ``pos``       -- ``(M, 3)`` positions.
    ``directors`` -- optional ``(M, 3)`` unit vectors drawn as arrows.
    ``color``     -- optional ``(M,)`` scalar per particle for the colormap
                     (defaults to the particle index, i.e. contour position).
    ``connect``   -- if True, draw a backbone line through consecutive positions.
    ``every``     -- subsample stride for the arrows (e.g. ``15`` -> one per 15).
    ``clim``      -- optional ``(lo, hi)`` colour limits.
    ``colorbar``  -- draw a colorbar (set False when sharing one across panels).

    Returns the scatter mappable (use ``sc.axes.figure`` for the figure).
    """
    pos = np.asarray(pos, float)
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    c = np.arange(len(pos)) if color is None else np.asarray(color, float)
    if connect:
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", color="0.7", lw=0.8, zorder=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, cmap=cmap, s=8, zorder=2)
    if clim is not None:
        sc.set_clim(*clim)
    if directors is not None:
        d = np.asarray(directors, float)
        idx = np.arange(0, len(pos), max(1, int(every)))
        ax.quiver(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                  d[idx, 0], d[idx, 1], d[idx, 2],
                  length=arrow_len, color=arrow_color, linewidth=1.2,
                  arrow_length_ratio=0.4, zorder=3)
    _equal_aspect(ax, pos)
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    if title:
        ax.set_title(title)
    if colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
        if clabel:
            cb.set_label(clabel)
    return sc
