import numpy as np
from typing import Tuple

from abc import ABCMeta, abstractmethod


class Projection(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def inverse(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
