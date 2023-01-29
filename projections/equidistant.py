import numpy as np

from typing import Tuple

from .base import Projection


class EquidistantProjection(Projection):
    """ Equidistant projection that is perfectly aligned to zenith-north """
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.sqrt(np.square(x) + np.square(y))
        a = np.arctan2(y, x)
        return z, a

    def invert(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return z * np.sin(a), z * np.cos(a)
