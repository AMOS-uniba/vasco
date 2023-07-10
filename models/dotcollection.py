import copy
import numpy as np

from projections import Projection
from photometry import Calibration


class DotCollection:
    def __init__(self, xy=None, i=None, mask=None, fnos=None):
        self._xy = np.empty(shape=(0, 2), dtype=float) if xy is None else xy
        self._i = np.empty(shape=(0,), dtype=float) if i is None else i
        self._fnos = np.zeros_like(i, dtype=float) if fnos is None else fnos
        self.mask = np.ones_like(self.x, dtype=bool) if mask is None else mask
        assert (xy is None) == (i is None), "Both or neither of xy and i must be set"
        assert self._xy.shape[0] == self._i.shape[0], "xy must be of shape (N, 2) and m of shape (N,)"
        assert self._xy.shape[0] == self._mask.shape[0], \
            f"xy must be of shape (N, 2) and is {self._xy.shape} and mask of shape (N,), is {self._mask.shape}"

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, pos=None):
        self._xy = np.empty(shape=(0, 2), dtype=float) if pos is None else pos

    @property
    def x(self):
        return self._xy[:, 0]

    @property
    def y(self):
        return self._xy[:, 1]

    @property
    def i(self):
        return self._i

    @property
    def count(self):
        return self._xy.shape[0]

    @property
    def count_valid(self):
        return np.count_nonzero(self.mask)

    def xs(self, masked):
        return self.x[self.mask] if masked else self.x

    def ys(self, masked):
        return self.y[self.mask] if masked else self.y

    def intensities(self, masked):
        return self.i[self.mask] if masked else self.i

    def fnos(self, masked):
        return self._fnos[self.mask] if masked else self._fnos

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m=None):
        self._mask = np.ones_like(self.i, dtype=bool) if m is None else ~m
        assert self.mask.shape == self.x.shape

    def culled_copy(self):
        out = copy.deepcopy(self)
        return out.cull()

    def cull(self):
        self._xy = self._xy[self.mask]
        self._i = self.i[self.mask]
        self.mask = None
        return self

    def project(self, projection: Projection, *, masked: bool) -> np.ndarray[float]:
        """ Project the dot collection from sensor to sky """
        return np.stack(projection(self.xs(masked), self.ys(masked)), axis=1)

    def calibrate(self, calibration: Calibration, *, masked: bool) -> np.ndarray[float]:
        return calibration(self.intensities(masked))
