import logging
import math

import dotmap
import numpy as np

from .base import Matcher

from astropy.coordinates import AltAz

from amosutils.projections import Projection
from amosutils.catalogue import Catalogue

from models import SensorData
from photometry import Calibration
from correctors import KernelSmoother
from correctors import kernels
from utilities import spherical_distance, spherical_difference, \
                      disk_to_altaz, altaz_to_disk, proj_to_disk, unit_grid

log = logging.getLogger('vasco')


class Counsellosr(Matcher):
    """
    The Counsellor is a Matcher that attempts to reconcile the sensor
    with the catalogue *after* the stars were paired to sensor dots.
    """

    def __init__(self, location, time, projection, *,
                 catalogue: Catalogue,
                 sensor_data: SensorData):
        super().__init__(location, time, projection.__class__)
        self.catalogue = catalogue
        self.sensor_data = sensor_data

        log.info(f"Counsellor created from")
        log.info(" - " + self.catalogue.__str__())
        log.info(" - " + self.sensor_data.__str__())

        errors = self.distance_sky(projection, mask_catalogue=False, mask_sensor=False)
        nearest = self.find_nearest_index(errors, axis=1)

    @staticmethod
    def compute_vector_errors(observed, catalogue):
        """
        Returns
        np.ndarray(N, 2): vector errors
        """
        catalogue = np.radians(catalogue)
        observed[..., 0] = math.tau / 4 - observed[..., 0]   # Convert observed co-altitude to altitude
        return spherical_difference(observed, catalogue)
        # BROKEN

    def position_errors(self, projection: Projection,
                        *,
                        mask_catalogue: bool, mask_sensor: bool):
        masked = mask_sensor or mask_catalogue
        sensor = self.sensor_data.stars.project(projection, masked=masked)
        altaz = self._altaz_cache \
            if self._altaz_cache is not None and self._altaz_cache.shape == sensor.shape \
            else self.catalogue.to_altaz(self.location, self.time, masked=masked)
        assert sensor.shape == altaz.shape, \
            f"The shape of the dot collection and the catalogue must be the same, got {sensor.shape} and {altaz.shape}"

        return self.compute_distances(sensor, altaz)
