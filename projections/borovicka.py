import numpy as np
import dotmap
import yaml
from typing import Tuple, Union

from .base import Projection
from .shifters import TiltShifter
from .transformers import BiexponentialTransformer
from .zenith import ZenithShifter


class BorovickaProjection(Projection):
    bounds = np.array((
        (None, None),  # x0
        (None, None),  # y0
        (None, None),  # a0
        (None, None),  # A
        (None, None),  # F
        (0.001, None), # V
        (None, None),  # S
        (None, None),  # D
        (None, None),  # P
        (None, None), # Q
        (0, None),     # epsilon
        (None, None),  # E
    ))

    def __init__(self,
                 x0: float = 0, y0: float = 0, a0: float = 0,
                 A: float = 0, F: float = 0,
                 V: float = 1, S: float = 0, D: float = 0, P: float = 0, Q: float = 0,
                 epsilon: float = 0, E: float = 0):

        assert V > 0, "Radial linear scale V must be > 0"

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
        return f"Borovička projection with \n" \
               f"   {self.axis_shifter} \n" \
               f"   {self.radial_transform} \n" \
               f"   {self.zenith_shifter}"

    def as_dict(self):
        return self.axis_shifter.as_dict() | self.radial_transform.as_dict() | self.zenith_shifter.as_dict()

    def as_tuple(self):
        return (
            self.axis_shifter.x0, self.axis_shifter.y0, self.axis_shifter.a0,
            self.axis_shifter.A, self.axis_shifter.F,
            self.radial_transform.linear, self.radial_transform.lin_coef, self.radial_transform.lin_exp,
            self.radial_transform.quad_coef, self.radial_transform.quad_exp,
            self.zenith_shifter.epsilon, self.zenith_shifter.E,
        )

    @staticmethod
    def load(file):
        data = dotmap.DotMap(yaml.safe_load(file), _dynamic=False)
        data = data.projection.parameters
        return BorovickaProjection(
            data.x0, data.y0, data.a0,
            data.A, data.F,
            data.V, data.S, data.D, data.P, data.Q,
            data.epsilon, data.E,
        )
