import logging
import math

import dotmap
import numpy as np

from .base import Matcher

from astropy.coordinates import AltAz

from models import Catalogue, SensorData
from projections import Projection
from photometry import Calibration
from correctors import KernelSmoother
from correctors import kernels
from utilities import spherical_distance, spherical_difference, \
                      disk_to_altaz, altaz_to_disk, proj_to_disk, unit_grid

log = logging.getLogger('vasco')


class Counselor(Matcher):
    """
    The Counselor is a Matcher that attempts to reconcile the sensor
    with the catalogue *after* the stars were paired to sensor dots.
    """

    def __init__(self, location, time, projection_cls, *,
                 catalogue: Catalogue,
                 sensor_data: SensorData):
        super().__init__(location, time, projection_cls)
        # a Counselor has fixed pairs: they have to be set on creation
        assert sensor_data.stars.count == catalogue.count, \
            f"Sensor data count ({sensor_data.stars.count}) does not match the catalogue data count ({catalogue.count})"
        self.catalogue = catalogue
        self.sensor_data = sensor_data
        self.position_smoother = None
        self.magnitude_smoother = None

        log.info(f"Counselor created with {self.catalogue.count} pairs:")
        log.info(" - " + self.catalogue.__str__())
        log.info(" - " + self.sensor_data.__str__())

    @property
    def count(self):
        return self.catalogue.count

    def mask_catalogue(self, mask):
        """ Here both methods mask_catalogue and mask_sensor_data must do both things """
        self.catalogue.set_mask(mask)
        self.sensor_data.set_mask(mask)

    def mask_sensor_data(self, mask):
        """ Here both methods are the same so just call the other one. """
        self.mask_catalogue(mask)

    @staticmethod
    def compute_distances(observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compute distance matrix for observed points projected to the sky and catalogue stars
        observed:   np.ndarray(N, 2)
        catalogue:  np.ndarray(N, 2)

        Returns
        -------
        np.ndarray(N): spherical distance between the dot and the associated star
        """
        observed[..., 0] = math.tau / 4 - observed[..., 0]   # Convert observed co-altitude to altitude
        return spherical_distance(observed, catalogue)

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

    def position_errors(self, projection: Projection, *, masked: bool):
        sensor = self.sensor_data.stars.project(projection, masked=masked)
        altaz = self._altaz \
            if self._altaz is not None and self._altaz.shape == sensor.shape \
            else self.catalogue.to_altaz(self.location, self.time, masked=masked)
        assert sensor.shape == altaz.shape, \
            f"The shape of the dot collection and the catalogue must be the same, got {sensor.shape} and {altaz.shape}"

        return self.compute_distances(sensor, altaz)

    def magnitude_errors(self, projection: Projection, calibration: Calibration, *, masked: bool):
        obs = calibration(self.sensor_data.stars.intensities(masked=masked))
        cat = self.catalogue.vmag(masked=masked)
        assert obs.shape == cat.shape, \
            f"The shape of the dot collection and the catalogue must be the same, got {obs.shape} and {cat.shape}"
        return obs - cat

    def position_errors_inverse(self, projection: Projection, *, masked: bool):
        return self.position_errors(projection, masked=masked)

    def pair(self, projection):
        self.catalogue.cull()
        self.sensor_data.stars_pixels.cull()
        return self

    def update_position_smoother(self, projection: Projection, *, bandwidth: float = 0.1):
        obs = proj_to_disk(self.sensor_data.stars.project(projection, masked=True))
        cat = altaz_to_disk(self.catalogue.altaz(self.location, self.time, masked=True))
        self.position_smoother = KernelSmoother(
            obs, obs - cat,
            kernel=kernels.nexp,
            bandwidth=bandwidth
        )

    def update_magnitude_smoother(self, projection: Projection, calibration: Calibration, *, bandwidth: float = 0.1):
        obs = proj_to_disk(self.sensor_data.stars.project(projection, masked=True))
        mcat = self.catalogue.vmag(masked=True)
        mobs = calibration(self.sensor_data.stars.intensities(masked=True))
        self.magnitude_smoother = KernelSmoother(
            obs, np.expand_dims(mobs - mcat, 1),
            kernel=kernels.nexp,
            bandwidth=bandwidth
        )

    def _meteor_xy(self, projection):
        """ Return on-disk xy coordinates for the meteor after applying the projection """
        return proj_to_disk(self.sensor_data.meteor.project(projection, masked=False))

    def project_meteor(self, projection: Projection):
        return disk_to_altaz(self._meteor_xy(projection))

    def correction_meteor_xy(self, projection: Projection):
        return self.position_smoother(self._meteor_xy(projection))

    def correction_meteor_mag(self, projection: Projection):
        return np.ravel(self.magnitude_smoother(self._meteor_xy(projection)))

    def correct_meteor_position(self, projection: Projection) -> AltAz:
        return disk_to_altaz(self._meteor_xy(projection) - self.correction_meteor_xy(projection))

    def correct_meteor_magnitude(self, projection: Projection, calibration: Calibration) -> np.ndarray[float]:
        return calibration(self.sensor_data.meteor.intensities(masked=False)) - self.correction_meteor_mag(projection)

    @staticmethod
    def _grid(smoother, resolution=21, *, masked: bool):
        xx, yy = unit_grid(resolution, masked=masked)
        nodes = np.ma.stack((xx.ravel(), yy.ravel()), axis=1)
        return smoother(nodes).reshape(resolution, resolution, -1)

    def position_grid(self, resolution=21):
        return self._grid(self.position_smoother, resolution, masked=True)

    def magnitude_grid(self, resolution=21):
        return self._grid(self.magnitude_smoother, resolution, masked=False)

    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        log.debug("Correcting a meteor")
        positions_raw = self.project_meteor(projection)
        positions_corrected = self.correct_meteor_position(projection)
        positions_correction_angle = positions_raw.separation(positions_corrected)
        positions_correction_xy = self.correction_meteor_xy(projection)

        intensities_raw = self.sensor_data.meteor.intensities(masked=False)
        intensities_corrected = self.correct_meteor_magnitude(projection, calibration)
        magnitudes_raw = calibration(intensities_raw)
        magnitudes_corrected = intensities_corrected
        magnitudes_correction = self.correction_meteor_mag(projection)

        return dotmap.DotMap(
            position_raw=positions_raw,
            position_corrected=positions_corrected,
            magnitudes_raw=magnitudes_raw,
            magnitudes_corrected=magnitudes_corrected,
            positions_correction_xy=positions_correction_xy,
            positions_correction_angle=positions_correction_angle,
            magnitudes_correction=magnitudes_correction,
            count=self.sensor_data.meteor.count,
            fnos=self.sensor_data.meteor.fnos(masked=False),
            _dynamic=False,
        )
