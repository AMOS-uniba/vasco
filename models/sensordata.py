import dotmap
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from utilities import polar_to_cart


class SensorData():
    """ A set of stars in xy format """

    def __init__(self):
        self.rect = dotmap.DotMap(left=-1, right=1, bottom=-1, top=1)
        self.points = np.empty(shape=(0, 2))
        self.intensities = np.empty(shape=(0,))
        self.count = 0

    def load(self, data):
        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = dotmap.DotMap(dict(left=0, top=0, right=w, bottom=h))
        self.points = np.asarray([[star.x, star.y] for star in data.Refstars])
        self.intensities = np.asarray([star.intensity for star in data.Refstars])
        self.use = np.ones_like(self.points[:, 0], dtype=bool)
        self.count = len(self.points)

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def m(self):
        return self.intensities

    @property
    def valid(self):
        return self.points[self.use]

    @property
    def xv(self):
        return self.points[self.use][:, 0]

    @property
    def yv(self):
        return self.points[self.use][:, 1]

    def project(self, projection, masked):
        if masked:
            return np.stack(projection(self.xv, self.yv), axis=0)
        else:
            return np.stack(projection(self.x, self.y), axis=0)

