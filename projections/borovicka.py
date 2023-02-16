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
        print(f"x = {x}, y = {y}")
        r, b = self.axis_shifter(x, y)
        print(f"r = {r}, b = {b}")
        u = self.radial_transform(r)
        print(f"u = {u}, b = {b}")
        z, a = self.zenith_shifter(u, b)
        print(f"z = {z}, a = {a}")
        return z, a

    def invert(self, z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"z = {z}, a = {a}")
        u, b = self.zenith_shifter.invert(z, a)
        print(f"u = {u}, b = {b}")
        r = self.radial_transform.invert(u)
        print(f"r = {r}, b = {b}")
        x, y = self.axis_shifter.invert(r, b)
        print(f"x = {x}, y = {y}")
        return x, y

    def __str__(self):
        return f"Borovička projection with \n" \
               f"   {self.axis_shifter} \n" \
               f"   {self.radial_transform} \n" \
               f"   {self.zenith_shifter}"
