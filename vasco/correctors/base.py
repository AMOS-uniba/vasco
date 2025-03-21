from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import NDArray

from amosutils.metrics import euclidean


class BaseCorrector(ABC):
    """
    A Corrector takes a bunch of reference points and their values (both possibly vectors)
    and constructs an evaluable function that can compute the estimate of the value
    at any point in the Euclidean space.
    """

    def __init__(self, points: NDArray, values: NDArray,
                 *,
                 metric: Callable[[NDArray, NDArray], NDArray] = euclidean):
        """ Store the points and associated values.

        Parameters
        ----------
        points : NDArray[N, ...] an array of N points in P dimensions
        values : NDArray[N, ...] an array of N values in Q dimensions
        """
        self.points = points
        self.values = values
        self.metric = metric

    @abstractmethod
    def __call__(self, nodes) -> NDArray:
        """ Estimate values of the function at nodes. """
