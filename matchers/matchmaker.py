import numpy as np

from typing import Optional, Callable

from .base import Matcher
from .counselor import Counselor

from projections import Projection, BorovickaProjection
from photometry import Calibration
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
        super().__init__(location, time, projection_cls)
        self.sensor_data = SensorData()

    def load_catalogue(self, filename: str):
        self.catalogue = Catalogue()
        self.catalogue.load(filename)

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData(filename)

    def update(self, location, time):
        super().update(location, time)

    def update_position_smoother(self, projection, **kwargs):
        """ There is no position smoother in Matchmaker """

    def mask_catalogue(self, mask):
        self.catalogue.set_mask(mask)

    def mask_sensor_data(self, mask):
        self.sensor_data.stars.mask = mask

    def _cartesian(self, func: Callable, projection: Projection, masked: bool, axis: int) -> np.ndarray:
        """
        Apply a function func over the Cartesian product of projected and catalogue stars
        and aggregate over the specified axis
        """
        return func(
            self.sensor_data.stars.project(projection, masked=masked),
            self.catalogue.to_altaz_deg(self.location, self.time, masked=masked),
            axis=axis,
        )

    def position_errors(self, projection: Projection, *, masked: bool) -> np.ndarray:
        return self._cartesian(self.find_nearest_value, projection, masked, 1)

    def errors_inverse(self, projection: Projection, *, masked: bool) -> np.ndarray:
        return self._cartesian(self.find_nearest_value, projection, masked, 0)

    def magnitude_errors(self,
                         projection: Projection,
                         calibration: Calibration,
                         *, masked: bool) -> np.ndarray:
        # Find which star is the nearest for every dot
        nearest = self._cartesian(self.find_nearest_index, projection, masked, 1)
        # Filter the catalogue by that index
        obs = calibration(self.sensor_data.stars.intensities(masked=masked))
        cat = self.catalogue.valid.iloc[nearest].vmag.values
        return obs - cat

    @staticmethod
    def compute_distances(observed, catalogue):
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
        Find the nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        return np.min(dist, axis=axis)

    def find_nearest_index(self, observed, catalogue, *, axis):
        """
        Find the index of the nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        return np.argmin(dist, axis=axis)

    def pair(self, projection: Projection) -> Counselor:
        # Find which star is the nearest for every dot
        nearest = self._cartesian(self.find_nearest_index, projection, masked=True, axis=1)
        # Filter the catalogue by that index
        cat = self.catalogue.valid.iloc[nearest]

        catalogue = Catalogue(cat[['dec', 'ra', 'vmag']])
        sensor_data = self.sensor_data.culled_copy()
        return Counselor(self.location, self.time, self.projection_cls, catalogue, sensor_data)
