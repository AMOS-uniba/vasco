import copy
import numpy as np


class DotCollection:
    def __init__(self, xy=None, m=None, mask=None):
        self.xy = xy
        self._m = np.empty(shape=(0,), dtype=float) if m is None else m
        self.mask = mask
        assert (xy is None) == (m is None), "Both or neither of xy and m must be set"
        assert self._xy.shape[0] == self._m.shape[0], "xy must be of shape (N, 2) and m of shape (N,)"
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
    def m(self):
        return self._m

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

    def ms(self, masked):
        return self.m[self.mask] if masked else self.m

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m=None):
        self._mask = np.ones_like(self.m, dtype=bool) if m is None else ~m
        assert self.mask.shape == self.x.shape

    def culled_copy(self):
        out = copy.deepcopy(self)
        return out.cull()

    def cull(self):
        self._xy = self._xy[self.mask]
        self._m = self.m[self.mask]
        self.mask = None
        return self

    def project(self, projection, *, masked: bool = False):
        return np.stack(projection(self.xs(masked), self.ys(masked)), axis=1)
