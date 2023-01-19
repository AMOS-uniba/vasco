import dotmap
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from utilities import polar_to_cart


class SensorData():
    """ A set of stars in xy format """

    def __init__(self, positions=None, intensities=None):
        self.rect = dotmap.DotMap(left=-1, right=1, bottom=-1, top=1)
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
        self.rect = dotmap.DotMap(dict(left=0, top=0, right=w, bottom=h))

        self.data = pd.DataFrame()
        self.data['x'] = np.asarray([star.x for star in data.Refstars])
        self.data['y'] = np.asarray([star.y for star in data.Refstars])
        self.data['i'] = np.asarray([star.intensity for star in data.Refstars])

        self.positions = np.asarray([[star.x, star.y] for star in data.Refstars])
        self.intensities = np.asarray([star.intensity for star in data.Refstars])
        self._count = self.positions.shape[0]
        self.reset_mask()

    def set_mask(self, mask):
        self.use = mask

    def reset_mask(self):
        self.use = np.ones_like(self.x, dtype=bool)
        print(f"Sensor data mask reset: {self.count_valid} / {self.count} stars used")

    def cull(self):
        self.positions = self.positions[self.use]
        self.intensities = self.intensities[self.use]
        self._count = 0
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

