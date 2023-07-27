import logging
import math
import dotmap
import numpy as np

from typing import Callable

from .base import Matcher
from .counselor import Counselor

from projections import Projection, BorovickaProjection
from photometry import Calibration
from models import SensorData, Catalogue
from utilities import spherical_distance

log = logging.getLogger('vasco')


class Matchmaker(Matcher):
    """
    Initial star matcher: loads a catalogue (dec/ra) and a set of points (x/y)
    and tries to find the best possible global match at location (lat/lon) and time (t)
    using a pre-defined projection class `projection_cls`
    """

    def __init__(self, location, time, projection_cls=BorovickaProjection, **kwargs):
        super().__init__(location, time, projection_cls, **kwargs)

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData.load_YAML(filename)

    def update(self, location, time):
        super().update(location, time)

    def update_position_smoother(self, projection, **kwargs):
        """ There is no position smoother in Matchmaker, can be implemented by child classes """

    def mask_catalogue(self, mask):
        self.catalogue.set_mask(mask)

    def mask_sensor_data(self, mask):
        self.sensor_data.set_mask(mask)

    def _cartesian(self, func: Callable, projection: Projection, masked: bool, axis: int) -> np.ndarray:
        """
        Apply a function func over the Cartesian product of projected and catalogue stars
        and aggregate over the specified axis
        """
        return func(
            self.sensor_data.stars.project(projection, masked=masked),
            self._altaz if self._altaz is not None else self.catalogue.to_altaz(self.location,
                                                                                self.time,
                                                                                masked=masked),
            axis=axis,
        )

    def position_errors(self, projection: Projection, *, masked: bool) -> np.ndarray:
        return self._cartesian(self.find_nearest_value, projection, masked, 1)

    def position_errors_inverse(self, projection: Projection, *, masked: bool) -> np.ndarray:
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
        if cat.size == 0:
            cat = np.tile(np.nan, obs.shape)
        if obs.size == 0:
            obs = np.tile(np.nan, cat.shape)
        return obs - cat

    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        raise NotImplementedError("Matchmaker cannot correct a meteor, use a Counselor instead")

    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
        raise NotImplementedError("Matchmaker cannot print corrected meteors, use a Counselor instead")

    @staticmethod
    def compute_distances(observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float, float]:
        """
        Compute distance matrix for observed points projected to the sky and catalogue stars

        Parameters
        ----------
        observed:   np.ndarray(M, 2)
        catalogue:  np.ndarray(N, 2)

        Returns
        -------
        np.ndarray(M, N)
        """
        observed = np.expand_dims(observed, 1)
        catalogue = np.expand_dims(catalogue, 0)
        observed[..., 0] = math.tau / 4 - observed[..., 0]   # Convert observed altitude to zenith distance
        return spherical_distance(observed, catalogue)

    def compute_vector_errors(self, observed, catalogue):
        """
        Compute vector errors for observed points projected onto the sky and catalogue stars

        Returns
        np.ndarray(M, N, 2)
        """
        raise NotImplementedError

    def find_nearest_value(self, observed, catalogue, *, axis):
        """
        Find the nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        return np.min(dist, axis=axis, initial=np.inf)

    def find_nearest_index(self, observed, catalogue, *, axis):
        """
        Find the index of the nearest dot to star or vice versa

        axis: int
            0 for nearest star to every dot
            1 for nearest dot to every star
        """
        dist = self.compute_distances(observed, catalogue)
        if dist.size > 0:
            return np.argmin(dist, axis=axis)
        else:
            return np.empty(shape=((observed.shape[0], catalogue.shape[0])[axis],), dtype=float)

    def pair(self, projection: Projection) -> Counselor:
        # Find which star is the nearest for every dot
        nearest = self._cartesian(self.find_nearest_index, projection, masked=True, axis=1)
        # Filter the catalogue by that index
        cat = self.catalogue.valid.iloc[nearest]

        catalogue = Catalogue(cat[['dec', 'ra', 'vmag']], name=self.catalogue.name)
        sensor_data = self.sensor_data.culled_copy()
        return Counselor(self.location, self.time, self.projection_cls, catalogue=catalogue, sensor_data=sensor_data)
