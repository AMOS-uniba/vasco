import logging
import math

import dotmap
import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from amosutils.projections import Projection, BorovickaProjection

from .base import Matcher
from .counsellor import Counsellor

from photometry import Calibration
from models import SensorData
from utilities import spherical_distance

log = logging.getLogger('vasco')


class Matchmaker(Matcher):
    """
    Initial star matcher: loads a catalogue (dec/ra) and a set of points (x/y)
    and tries to find the best possible global match at location (lat/lon) and time (t)
    using a pre-defined projection class `projection_cls`
    """

    def __init__(self,
                 location: EarthLocation,
                 time: Time,
                 projection_cls: type[Projection] = BorovickaProjection,
                 **kwargs):
        super().__init__(location, time, projection_cls, **kwargs)

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData.load_YAML(filename)

    def update_position_smoother(self, projection, **kwargs):
        """ There is no position smoother in Matchmaker, can be implemented by child classes """

    def mask_catalogue(self, mask):
        self.catalogue.mask &= mask
        self.invalidate_altaz()

    def mask_sensor_data(self, mask):
        self.sensor_data._stars.mask &= mask

    def position_errors_sensor_star_to_dot(self,
                                           projection: Projection,
                                           *,
                                           masked: bool) -> np.ndarray[float]:
        """
        Find errors from every star to the nearest dot, on sensor
        Return a numpy array (N): distance [µm]
        """

    def position_errors_sensor_dot_to_star(self,
                                           projection: Projection,
                                           *,
                                           masked: bool) -> np.ndarray[float]:
        """
        Find errors from every dot to the nearest star, on sensor
        Return a numpy array (N): distance [µm]
        """

    def magnitude_errors_sky(self,
                             projection: Projection,
                             calibration: Calibration,
                             axis: int,
                             *,
                             mask_catalogue: bool,
                             mask_sensor: bool) -> np.ndarray:
        # Find which star is the nearest for every dot
        nearest = self.find_nearest_index(
            self.distance_sky(projection, mask_catalogue=mask_catalogue, mask_sensor=mask_sensor),
            axis=axis
        )
        obs = calibration(self.sensor_data.stars.intensities(masked=mask_sensor))
        # Filter the catalogue by that index
        cat = self.catalogue.vmag(self.location, self.time, masked=mask_catalogue)[nearest]

        if cat.size == 0:
            cat = np.tile(np.nan, obs.shape)
        if obs.size == 0:
            obs = np.tile(np.nan, cat.shape)

        return obs - cat

    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        raise NotImplementedError("Matchmaker cannot correct a meteor, use a Counsellor instead")

    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
        raise NotImplementedError("Matchmaker cannot print corrected meteors, use a Counsellor instead")

    def _compute_distances_sky(self, observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float]:
        self._compute_distances_sky_cart(observed, catalogue)

    @staticmethod
    def _compute_distances_sensor(observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float]:
        observed = np.expand_dims(observed, 1)
        catalogue = np.expand_dims(catalogue, 0)
        #return distance(observed, catalogue)

    @staticmethod
    def compute_vector_errors(observed, catalogue):
        """
        Compute vector errors for observed points projected onto the sky and catalogue stars

        Returns
        np.ndarray(M, N, 2)
        """
        raise NotImplementedError

    def pair(self, projection: Projection) -> Counsellor:
        return Counsellor(self.location, self.time, projection,
                          catalogue=self.catalogue, sensor_data=self.sensor_data)
