"""
Choosing a kernel smoother's bandwidth, rather than declaring one.

The smoother is Nadaraya-Watson: the correction at a point is the weighted mean of the residuals of
the stars around it, each weighted by the kernel of its distance over the bandwidth. So the
bandwidth is the only thing that decides whether the correction follows the field or the noise --
too small and every star gets its own residual back and the meteor between them gets whatever star
happens to be nearest; too large and every point gets the same global mean, which is barely a
correction at all.

**In-sample error cannot choose it.** Every star sits at distance zero from itself, where the
kernel is at its largest, so a smaller bandwidth always reproduces the training residuals better
and the minimum is at zero, which predicts nothing. Leave-one-out is what settles it, and for this
estimator it needs no refitting: dropping a point from a weighted mean is zeroing its weight and
renormalising, so one distance matrix serves every candidate bandwidth.

What this measures is how well the field predicts a star it has not seen. That is the right
question and not quite the question asked -- the meteor is not a star, and the correction is read
where the meteor is, which may be further from any star than a star typically is from its
neighbours. Whoever wants to do better than this has to say what "typical" means for a trail.
"""
import logging
from collections.abc import Callable

import numpy as np
from demeteor.metrics import euclidean
from numpy.typing import NDArray

from . import kernels

log = logging.getLogger('vasco')

#: The range searched when nothing else is said, in units of the projection disk whose radius is
#: one -- so 0.005 is about half a degree of zenith distance and 2 is the whole sky twice over.
#: Geometric, because what matters about a bandwidth is its order of magnitude.
DEFAULT_MIN = 0.005
DEFAULT_MAX = 2.0
DEFAULT_STEPS = 25

#: Below this many points there is nothing to cross-validate against and the answer would be noise
MIN_POINTS = 6


def loo_score(points: NDArray,
              values: NDArray,
              bandwidth: float,
              *,
              kernel: Callable = kernels.nexp,
              metric: Callable = euclidean) -> float:
    """
    Mean squared leave-one-out residual of the smoother at this bandwidth.

    Parameters
    ----------
    points:     NDArray(N, 2) where the residuals were measured
    values:     NDArray(N, D) what they were -- two components for a position, one for a magnitude

    Returns
    -------
    The mean over points of the squared distance between a point's own value and what the smoother
    built from every *other* point predicts there. Infinite if it cannot be computed.
    """
    if points.shape[0] < MIN_POINTS:
        return np.inf

    distances = metric(np.expand_dims(points, 1), np.expand_dims(points, 0))
    weights = kernel(distances / bandwidth)
    # The whole of leave-one-out, in one line: a point does not vote on itself.
    np.fill_diagonal(weights, 0.0)

    total = np.sum(weights, axis=0)
    # A point every other point is too far from to weigh at all -- which happens at a small
    # bandwidth -- has nothing to be predicted from, and is left out rather than made a nan.
    usable = np.isfinite(total) & (total > 0)
    if not np.any(usable):
        return np.inf

    predicted = (weights.T @ values)[usable] / total[usable, None]
    residual = values[usable] - predicted
    if not np.all(np.isfinite(residual)):
        return np.inf

    return float(np.mean(np.sum(np.square(residual), axis=1)))


def select(points: NDArray,
           values: NDArray,
           *,
           minimum: float = DEFAULT_MIN,
           maximum: float = DEFAULT_MAX,
           steps: int = DEFAULT_STEPS,
           kernel: Callable = kernels.nexp,
           metric: Callable = euclidean) -> tuple[float, float, list[tuple[float, float]]]:
    """
    The bandwidth with the smallest leave-one-out error, over a geometric grid.

    A grid and not an optimiser: the curve has one minimum and a broad one, the whole search is a
    few matrix products, and a grid cannot wander off or fail to converge -- which matters when
    nobody is watching.

    Returns
    -------
    The chosen bandwidth, its score, and the whole curve, so that a caller can record how flat the
    minimum was rather than only where it fell.
    """
    grid = np.geomspace(minimum, maximum, steps)
    curve = [(float(bandwidth), loo_score(points, values, bandwidth, kernel=kernel, metric=metric))
             for bandwidth in grid]

    finite = [(bandwidth, score) for bandwidth, score in curve if np.isfinite(score)]
    if not finite:
        log.warning(f"No bandwidth in [{minimum}, {maximum}] could be scored over "
                    f"{points.shape[0]} points; falling back to {DEFAULT_MIN * 10}")
        return DEFAULT_MIN * 10, float('inf'), curve

    best, score = min(finite, key=lambda row: row[1])

    # Worth saying out loud rather than only returning: a minimum at the edge means the grid did
    # not contain the answer, and a bandwidth of the whole sky is not a correction.
    if best in (grid[0], grid[-1]):
        log.warning(f"The best bandwidth {best:.5f} is at the edge of the searched range "
                    f"[{minimum}, {maximum}] -- the answer may lie outside it")

    log.info(f"Chose bandwidth {best:.5f} over {points.shape[0]} points "
             f"(leave-one-out mse {score:.4e})")
    return best, score, curve
