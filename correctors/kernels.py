import numpy as np

from typing import Callable


def gaussian(x: np.ndarray[float], mu: float, sigma: float) -> np.ndarray[float]:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))


def nexp(x):
    return np.exp(-x)


def ugauss(x):
    return gaussian(x, 0, 1)


def inv(soften: float = 1e-6) -> Callable[[float], float]:
    def fun(x):
        return 1 / (x + soften)
    return fun


def epanechnikov(x: np.ndarray[float]) -> np.ndarray[float]:
    return np.where(x > 1, 0, 0.75 * (1 - x**2))
