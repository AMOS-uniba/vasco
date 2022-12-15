import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from typing import Optional

from .base import Comparator
from projections import BorovickaProjection
from models import SensorData, Catalogue
from utilities import spherical_distance


class StarMatcher(Comparator):
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
        self.update_sky()

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData(filename)

    def update(self, location, time):
        self.location = location
        self.time = time

        if self.catalogue is not None:
            self.update_sky()

    def update_sky(self):
        self.sky = self.catalogue.to_altaz(self.location, self.time)

    def avg_error(self, errors) -> float:
        return np.sqrt(np.sum(np.square(errors)) / self.sensor_data.count)

    def max_error(self, errors) -> float:
        return np.max(errors)

    def errors(self, projection) -> np.ndarray:
        return self.find_nearest_value(self.sensor_data.project(projection), self.sky)

    def compute_distances(self, observed, catalogue):
        observed = observed[observed[:, 0] < np.pi / 2]     # Cull stars that are below the horizon
        observed = np.expand_dims(observed, 1)
        catalogue = np.expand_dims(catalogue, 2)
        catalogue = np.radians(catalogue)
        observed[0, :, :] = np.pi / 2 - observed[0, :, :]   # Convert observed altitude to zenith distance
        return spherical_distance(observed, catalogue)

    def find_nearest_value(self, observed, catalogue):
        dist = self.compute_distances(observed, catalogue)
        return np.min(dist, axis=0)

    def find_nearest_index(self, observed, catalogue):
        dist = self.compute_distances(observed, catalogue)
        return np.argmin(dist, axis=0)

    def pair(self, projection):
        nearest = self.find_nearest_index(self.sensor_data.project(projection), self.sky)
        return np.take(self.sensor_data, nearest)

    def func(self, x):
        return self.avg_error(self.errors(self.projection_cls(*x)))

