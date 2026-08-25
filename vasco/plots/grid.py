"""
Lattices to sample a correction field on, for the quiver plots.

Choosing where to draw arrows is a drawing decision, so it lives here and not in the library that
computes the field. demetria hands back something callable; this decides where to call it.

**Two lattices, on purpose.** utilities.py had two functions both called `unit_grid`, the second
shadowing the first, and the shadowing one is what had been running. They are not variants of one
idea -- a square lattice and a triangular one look quite different under arrows, and which reads
better depends on the field -- so both are kept and named, and `LATTICES` is what a caller switches
between. `DEFAULT` is the triangular one, which is what the plots have been getting.

At resolution 21, masked to the disk: square gives 313 arrows, triangular 89.

Note that nothing draws these at present. `MainWindowPlots._plot_correction_grid` computes a lattice
and discards it, because the line that would have drawn it is commented out -- which is also why
nobody noticed there were two of these.
"""
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def square(resolution: int, *, masked: bool) -> tuple[NDArray, NDArray]:
    """
    A plain square lattice over the unit square, optionally clipped to the disk.

    What the name `unit_grid` implies, and the definition that was being shadowed.
    """
    s = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(s, s)

    if not masked:
        return x, y

    outside = x ** 2 + y ** 2 > 1
    return np.ma.masked_array(x, outside), np.ma.masked_array(y, outside)


def triangular(resolution: int, *, masked: bool) -> tuple[NDArray, NDArray]:
    """
    A triangular lattice: every point equidistant from its six neighbours.

    Built by shearing a square lattice -- squash y by sin(60 degrees), then slide each row half a
    step -- over [-2, 2] rather than [-1, 1], because the shear pushes a good deal of it outside the
    disk and this keeps the inside populated. Coarser than `square` at the same resolution, and
    generally easier to read under arrows, since no two arrows line up in a column.
    """
    s = np.linspace(-2, 2, resolution)
    x, y = np.meshgrid(s, s)
    y = y * np.sqrt(3) / 2
    x = x + y / 2

    if not masked:
        return x, y

    outside = x ** 2 + y ** 2 > 1
    return np.ma.masked_array(x, outside), np.ma.masked_array(y, outside)


#: What a caller switches between, by name.
LATTICES: dict[str, Callable[..., tuple[NDArray, NDArray]]] = {
    'square': square,
    'triangular': triangular,
}

#: The one that has been running.
DEFAULT = 'triangular'


def lattice(resolution: int, *, masked: bool, kind: str = DEFAULT) -> tuple[NDArray, NDArray]:
    """ The named lattice, or a KeyError naming the ones there are. """
    if kind not in LATTICES:
        raise KeyError(f"No lattice called {kind!r}; there are {sorted(LATTICES)}")
    return LATTICES[kind](resolution, masked=masked)


def sample(smoother, resolution: int = 21, *, masked: bool,
           kind: str = DEFAULT) -> NDArray:
    """
    A correction field evaluated on a lattice, shaped for imshow or quiver.

    Was Matcher._grid, and it is drawing rather than matching: the library gives you a field, and
    how densely to look at it is a question about the picture.
    """
    xx, yy = lattice(resolution, masked=masked, kind=kind)
    nodes = np.ma.stack((xx.ravel(), yy.ravel()), axis=1)
    return smoother(nodes).reshape(resolution, resolution, -1)
