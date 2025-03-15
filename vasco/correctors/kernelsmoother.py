
import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .base import BaseCorrector
from . import kernels
from .metric import spherical, euclidean


class KernelSmoother(BaseCorrector):
    def __init__(self,
                 points: NDArray,
                 values: NDArray,
                 *,
                 metric: Callable[[NDArray, NDArray], NDArray] = euclidean,
                 kernel=kernels.nexp,
                 bandwidth=1.0):
        super().__init__(points, values, metric=metric)
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, nodes):
        # Calculate Euclidean distance from every point to every node
        distances = self.metric(np.expand_dims(self.points, 1), np.expand_dims(nodes, 0))
        # Calculate influences as a kernel function of bandwidth-scaled distance
        infl = self.kernel(distances / self.bandwidth)
        # Calculate the sum of weighted votes
        votes = np.sum(np.expand_dims(infl, 2) * np.expand_dims(self.values, 1), axis=0)
        # Calculate the overall sum of weights for normalization
        sums = np.expand_dims(np.sum(infl, axis=0), 1)
        return votes / sums
