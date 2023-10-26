import numpy as np
from typing import Tuple

from abc import ABCMeta, abstractmethod


class Projection(metaclass=ABCMeta):
    """
    A base class for all projections. Should implement xy -> za and za -> xy conversions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply this projection to an array of points """

    @abstractmethod
    def invert(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply an inverse projection to an array of points """

    @abstractmethod
    def as_dict(self):
        """ Return a dict representation of the Projection's parameters """
        pass
