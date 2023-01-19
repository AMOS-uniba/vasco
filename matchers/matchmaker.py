import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from typing import Optional

from .base import Matcher
from .counselor import Counselor

from projections import BorovickaProjection
from models import SensorData, Catalogue
from utilities import spherical_distance


class Matchmaker(Matcher):
    """
    Initial star matchmaker: loads a catalogue (dec/ra) and a set of points (x/y)
    and tries to find the best possible global match at location (lat/lon) and time (t)
    using a pre-defined projection class `projection_cls`
    """

    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.catalogue = None
        self.sky = None
        super().__init__(location, time, projection_cls)
        self.sensor_data = SensorData()

    def load_catalogue(self, filename: str):
        self.catalogue = Catalogue()
        self.catalogue.load(filename)
        self.update_sky()

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData(filename)

    def update(self, location, time):
        super().update(location, time)
        if self.catalogue is not None:
            self.update_sky()

    @property
    def count(self):
        return self.sensor_data.count

    def cull_catalogue(self, mask):
        self.catalogue.set_mask(mask)

    def cull_sensor_data(self, mask):
        self.sensor_data.set_mask(mask)

    def errors(self, projection, masked) -> np.ndarray:
        return self.find_nearest_value(
            self.sensor_data.project(projection, masked=masked),
            self.sky,
            axis=1
        )

    def errors_inverse(self, projection, masked) -> np.ndarray:
        return self.find_nearest_value(
            self.sensor_data.project(projection, masked=masked),
            self.sky,
            axis=0
        )

    def compute_distances(self, observed, catalogue):
        """
        Compute distance matrix for observed points projected to the sky and catalogue stars
        observed:   np.ndarray(M, 2)
        catalogue:  np.ndarray(N, 2)

        Returns
        np.ndarray(M, N)
        """
        observed = np.expand_dims(observed, 1)
        catalogue = np.expand_dims(catalogue, 0)
        catalogue = np.radians(catalogue)
        observed[..., 0] = np.pi / 2 - observed[..., 0]   # Convert observed altitude to zenith distance
        return spherical_distance(observed, catalogue)

    def compute_vector_errors(self, observed, catalogue):
        """
        Compute vector errors for observed points projected onto the sky and catalogue stars

        Returns
        np.ndarray(M, N, 2)
        """
        observed = np.expand_dims(observed, 1)
        catalogue = np.expand_dims(catalogue, 0)
        catalogue = np.radians(catalogue)
        observed[..., 0] = np.pi / 2 - observed[..., 0]   # Convert observed altitude to zenith distance
        raise NotImplementedError

    def find_nearest_value(self, observed, catalogue, *, axis):
        """
        Find nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        return np.min(dist, axis=axis)

    def find_nearest_index(self, observed, catalogue, *, axis):
        """
        Find the index of nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        return np.argmin(dist, axis=axis)

    def pair(self, projection):
        # Find which star is the nearest for every dot
        nearest = self.find_nearest_index(self.sensor_data.project(projection, True), self.sky, axis=1)
        # Filter the catalogue by that index
        cat = self.catalogue.valid.iloc[nearest]

        sensor_data = SensorData(self.sensor_data.valid, self.sensor_data.intensities)
        catalogue = Catalogue(cat[['dec', 'ra', 'vmag']])
        return Counselor(self.location, self.time, self.projection_cls, catalogue, sensor_data)

    def func(self, x):
        return self.avg_error(self.errors(self.projection_cls(*x), True))
