import numpy as np
import scipy as sp
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from typing import Optional

from projections import BorovickaProjection

from models import SensorData, Catalogue

from utilities import spherical_distance


class StarMatcher():
    """
    Initial star matcher: loads a catalogue (dec/ra) and a set of points (x/y)
    and tries to find the best possible match at time (t) and location (lat/lon)
    """

    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.projection_cls = projection_cls
        self.catalogue = None
        self.sky = None
        self.update(location, time)

    def load_catalogue(self, filename: str):
        self.catalogue = Catalogue(filename)

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData(filename)

    def update(self, location, time):
        self.location = location
        self.time = time

        if self.catalogue is not None:
            self.update_sky()

    def update_sky(self):
        self.sky = self.catalogue.to_altaz(self.location, self.time)

    def mean_error(self, projection) -> float:
        return np.sqrt(np.sum(np.square(self.errors(projection))) / self.sensor_data.count)

    def max_error(self, projection) -> float:
        return np.max(self.errors(projection))

    def errors(self, projection) -> np.ndarray:
        return self.find_nearest_value(self.sensor_data.project(projection), self.sky)

    def compute_distance(self, stars, catalogue):
        stars = stars[stars[:, 0] < np.pi / 2]
        stars = np.expand_dims(stars, 1)
        catalogue = np.expand_dims(catalogue, 2)
        catalogue = np.radians(catalogue)
        stars[0, :, :] = np.pi / 2 - stars[0, :, :]
        print(stars.shape, catalogue.shape)
        return spherical_distance(stars, catalogue)

    def find_nearest_value(self, stars, catalogue):
        dist = self.compute_distance(stars, catalogue)
        nearest = np.min(dist, axis=0)
        return nearest

    def find_nearest_index(self, stars, catalogue):
        dist = self.compute_distance(stars, catalogue)
        nearest = np.argmin(dist, axis=0)
        return nearest

    def pair(self, projection):
        nearest = self.find_nearest_index(self.sensor_to_sky(projection), self.catalogue_altaz)
        return np.take(self.stars, nearest)

    def func(self, x):
        return self.mean_error(self.projection_cls(*x))

    def minimize(self, x0=(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0), maxiter=30):
        result = sp.optimize.minimize(self.func, x0, method='Nelder-Mead',
            bounds=((None, None), (None, None), (None, None), (None, None), (None, None), (0, None), (None, None), (None, None), (None, None), (None, None), (0, None), (None, None)),
            options=dict(maxiter=maxiter, disp=True),
        )
        return result
