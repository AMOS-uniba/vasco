import numpy as np
from typing import Tuple


class OpticalAxisShifter():
    """ Shifts and derotates the optical axis of the sensor """
    def __init__(self, *, x0: float=0, y0: float=0, a0: float=0, E: float=0):
        self.x0 = x0                # x position of the centre of optical axis
        self.y0 = y0                # y position of the centre of optical axis
        self.a0 = a0                # rotation of the optical axis
        self.E = E                  # azimuth of centre of FoV

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = x - self.x0
        ys = y - self.y0
        r = np.sqrt(np.square(xs) + np.square(ys))
        b = self.a0 - self.E + np.arctan2(ys, xs)
        return r, b

    def inverse(self, r: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi = b - self.a0 + self.E
        x = self.x0 + r * np.cos(xi)
        y = self.y0 + r * np.sin(xi)
        return x, y


class EllipticShifter(OpticalAxisShifter):
    def __init__(self, *, x0: float=0, y0: float=0, a0: float=0, A: float=0, F: float=0, E: float=0):
        super().__init__(x0=x0, y0=y0, a0=a0, E=E)
        self.A = A                  # elliptic stretch, amplitude
        self.F = F                  # elliptic stretch, phase

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r, b = super().__call__(x, y)
        r += self.A * (y - self.y0) * np.cos(self.F - self.a0) \
            - self.A * (x - self.x0) * np.sin(self.F - self.a0)
        return r, b

    def inverse(self, r: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Not implemented yet")
        pass
