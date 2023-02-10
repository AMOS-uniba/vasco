import numpy as np
from typing import Tuple, Union

from .base import Projection
from .shifters import TiltShifter
from .transformers import BiexponentialTransformer
from .zenith import ZenithShifter


class BorovickaProjection(Projection):
    def __init__(self, x0: float = 0, y0: float = 0, a0: float = 0,
                 A: float = 0, F: float = 0,
                 V: float = 1, S: float = 0, D: float = 0, P: float = 0, Q: float = 0,
                 epsilon: float = 0, E: float = 0):
        #        assert(epsilon >= 0 and V >= 0)
        super().__init__()
        self.axis_shifter = TiltShifter(x0=x0, y0=y0, a0=a0, A=A, F=F, E=E)
        self.radial_transform = BiexponentialTransformer(V, S, D, P, Q)
        self.zenith_shifter = ZenithShifter(epsilon=epsilon, E=E)

    def __call__(self,
                 x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        r, b = self.axis_shifter(x, y)
        u = self.radial_transform(r)
        z, a = self.zenith_shifter(u, b)
        return z, a

    def invert(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u, b = self.zenith_shifter.invert(z, a)
        r = self.radial_transform.invert(u)
        x, y = self.axis_shifter.invert(r, b)
        return x, y

    def __str__(self):
        return f"Borovička projection with " \
               f"x0 = {self.x0:.6f}, " \
               f"y0 = {self.y0:.6f}, " \
               f"a0 = {self.a0:.6f}, " \
               f"V = {self.V:.6f}, " \
               f"S = {self.S:.6f}, " \
               f"D = {self.D:.6f}, " \
               f"P = {self.P:.6f}, " \
               f"Q = {self.Q:.6f}, " \
               f"A = {self.A:.6f}, " \
               f"F = {self.F:.6f}, " \
               f"epsilon = {self.epsilon:.6f}, " \
               f"E = {self.E:.6f}"
