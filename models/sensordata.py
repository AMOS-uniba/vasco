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


class SensorData():
    """ A set of stars in xy format """

    def __init__(self, positions=None, intensities=None):
        self.rect = Rect(-1, 1, -1, 1)
        self.positions = np.empty(shape=(0, 2))
        self.intensities = np.empty(shape=(0,))
        self._count = 0

        if positions is not None:
            self.positions = positions
            self._count = self.positions.shape[0]
            if intensities is not None:
                self.intensities = intensities

        self.reset_mask()

    def load(self, data):
        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = Rect(0, w, 0, h)

        self.positions = np.asarray([[star.x, star.y] for star in data.Refstars])
        self.unitdisk = self.rect.to_unit(self.positions)

        self.intensities = np.asarray([star.intensity for star in data.Refstars])
        self._count = self.positions.shape[0]
        self.reset_mask()

    def set_mask(self, mask):
        self.use = ~mask

    def reset_mask(self):
        self.use = np.ones_like(self.x, dtype=bool)
        print(f"Sensor data mask reset: {self.count_valid} / {self.count} stars used")

    def culled_copy(self):
        out = copy.deepcopy(self)
        out.cull()
        return out

    def cull(self):
        self.positions = self.positions[self.use]
        self.intensities = self.intensities[self.use]
        self._count = self.positions.shape[0]
        self.reset_mask()

    @property
    def count(self):
        return self._count

    @property
    def count_valid(self):
        return np.count_nonzero(self.use)

    @property
    def xy(self):
        return self.positions

    @property
    def x(self):
        return self.positions[:, 0]

    @property
    def y(self):
        return self.positions[:, 1]

    @property
    def m(self):
        return self.intensities

    @property
    def valid(self):
        return self.positions[self.use]

    @property
    def valid_intensities(self):
        return self.intensities[self.use]

    @property
    def xv(self):
        return self.positions[self.use][:, 0]

    @property
    def yv(self):
        return self.positions[self.use][:, 1]

    @property
    def xyv(self):
        return self.points[self.use]

    def project(self, projection, masked=False):
        if masked:
            return np.stack(projection(self.xv, self.yv), axis=1)
        else:
            return np.stack(projection(self.x, self.y), axis=1)

    def to_disk(self, masked=False):
        if masked:
            return np.stack(self.rect.shifter(self.xv, self.yv), axis=1)
        else:
            return np.stack(self.rect.shifter(self.x, self.y), axis=1)

    def __str__(self):
        return f"<Sensor data of {self.count_valid} / {self.count} objects>"
