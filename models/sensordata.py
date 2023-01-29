import copy
import dotmap
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from projections.shifters import ScalingShifter
from utilities import polar_to_cart


class Rect():
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xcen = (xmax + xmin) / 2
        self.ycen = (ymax + ymin) / 2
        self.r = np.sqrt((ymax - ymin)**2 + (xmax - xmin)**2) / 2
        self.shifter = ScalingShifter(x0=self.xcen, y0=self.ycen, scale=1 / self.r)

    def to_unit(self, data):
        """ data: np.ndarray(N, 2) """
        out = np.zeros_like(data)
        out[:, 0] = (data[:, 0] - self.xcen) / self.r
        out[:, 1] = (data[:, 1] - self.ycen) / self.r
        return out


class DotCollection():
    def __init__(self, xy=None, m=None, mask=None):
        self._xy = np.empty(shape=(0, 2)) if xy is None else xy
        self._m = np.empty(shape=(0,)) if m is None else m
        self.mask = mask
        assert self._xy.shape[0] == self._m.shape[0], "xy must be of shape (N, 2) and m of shape (N,)"
        assert self._xy.shape[0] == self._mask.shape[0], "xy must be of shape (N, 2) and mask of shape (N,)"

    @property
    def xy(self):
        return self._xy

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
    def mask(self):
        return self._mask

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
        self._count = self._xy.shape[0]
        self.mask = None
        return self

    def project(self, projection, *, masked=False):
        return np.stack(projection(self.xs(masked), self.ys(masked)), axis=1)


class SensorData():
    """ A set of stars in xy format """

    def __init__(self, star_positions=None, star_intensities=None, meteor_positions=None, meteor_intensities=None):
        self.rect = Rect(-1, 1, -1, 1)
        self.stars = DotCollection(star_positions, star_intensities, None)
        self.meteor = DotCollection(meteor_positions, meteor_intensities, None)

    def load(self, data):
        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = Rect(0, w, 0, h)
        self.stars = DotCollection(
            np.asarray([[star.x, star.y] for star in data.Refstars]),
            np.asarray([star.intensity for star in data.Refstars]),
        )
        self.meteor = DotCollection(
            np.asarray([[snapshot.xc, snapshot.yc] for snapshot in data.Trail]),
            np.asarray([snapshot.intensity for snapshot in data.Trail]),
        )

    def stars_to_disk(self, masked):
        return np.stack(self.rect.shifter(self.stars.xs(masked), self.stars.ys.masked), axis=1)

    def meteor_to_disk(self, masked):
        return np.stack(self.rect.shifter(self.meteor.xs(masked), self.meteor.ys.masked), axis=1)

    def reset_mask(self):
        self.stars.mask = None

    def culled_copy(self):
        out = copy.deepcopy(self)
        out.stars.cull()
        return out

    def __str__(self):
        return f"<Sensor data with {self.stars.count_valid} / {self.stars.count} reference stars and {self.meteor.count} meteor snapshots>"
