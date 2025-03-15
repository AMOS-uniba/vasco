import logging
import math

import dotmap
import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from amosutils.projections import Projection, BorovickaProjection

from .base import Matcher

from photometry import Calibration

log = logging.getLogger('vasco')


class Mataschmaker(Matcher):
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

    def update_position_smoother(self, projection, **kwargs):
        """ There is no position smoother in Matchmaker, can be implemented by child classes """


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

    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
        raise NotImplementedError("Matchmaker cannot print corrected meteors, use a Counsellor instead")

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
