import numpy as np
import scipy as sp
from typing import Tuple


class ScalingShifter():
    """ Shifts and scales the sensor without rotation """
    def __init__(self, *, x0: float=0, y0: float=0, scale: float=0):
        self.x0 = x0
        self.y0 = y0
        self.scale = scale

    def __call__(self, x: np.ndarray, y: np.ndarray):
        xs = (x - self.x0) * self.scale
        ys = (y - self.y0) * self.scale
        return xs, ys


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

    def invert(self, r: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xi = b - self.a0 + self.E
        x = self.x0 + r * np.cos(xi)
        y = self.y0 + r * np.sin(xi)
        return x, y

    def __str__(self):
        return f"<{self.__class__} x0={self.x0} y0={self.y0} a0={self.a0} E={self.E}>"


class TiltShifter(OpticalAxisShifter):
    """ Extends OpticalAxisShifter with imaging plane tilt """
    def __init__(self, *, x0: float=0, y0: float=0, a0: float=0, A: float=0, F: float=0, E: float=0):
        super().__init__(x0=x0, y0=y0, a0=a0, E=E)
        self.A = A                  # tilt stretch, amplitude
        self.F = F                  # tilt stretch, phase
        self.cos_term = np.cos(self.F - self.a0)
        self.sin_term = np.sin(self.F - self.a0)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r, b = super().__call__(x, y)
        r += self.A * ((y - self.y0) * self.cos_term - (x - self.x0) * self.sin_term)
        return r, b

    def jacobian(self, vec, r, b) -> Tuple[np.ndarray, np.ndarray]:
        xs = vec[0] - self.x0
        ys = vec[1] - self.y0
        r2 = np.square(xs) + np.square(ys)
        drdx = xs / np.sqrt(r2) - self.A * self.sin_term
        drdy = ys / np.sqrt(r2) + self.A * self.cos_term
        dbdx = -ys / r2
        dbdy = xs / r2
        return [
            [drdx, drdy],
            [dbdx, dbdy],
        ]

    def func(self, vec, r, b):
        q = self.__call__(vec[0], vec[1])
        print(vec, r, b)
        return q[0] - r, q[1] - b

    def invert(self, r: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return sp.optimize.root(
            self.func,
            super().invert(r, b),
            args=(r, b),
            jac=self.jacobian,
            tol=1e-9,
        ).x

    def __str__(self):
        return f"<{self.__class__} x0={self.x0} y0={self.y0} a0={self.a0} A={self.A} F={self.F} E={self.E}>"

