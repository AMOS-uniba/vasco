import logging
import dotmap
import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from amosutils.projections import Projection, BorovickaProjection

from .base import Matcher
from .counsellor import Counsellor

from photometry import Calibration
from models import SensorData
from utilities import spherical, spherical_distance, hash_numpy

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
        self._cached_distances = None
        self._hash_observed = None
        self._hash_catalogue = None

    def load_sensor(self, filename: str):
        self.sensor_data = SensorData.load_YAML(filename)

    def update_position_smoother(self, projection, **kwargs):
        """ There is no position smoother in Matchmaker, can be implemented by child classes """

    def mask_catalogue(self, mask):
        self.catalogue.mask &= mask
        self.invalidate_altaz()

    def mask_sensor_data(self, mask):
        self.sensor_data.stars.mask &= mask

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

    def position_errors_sky(self,
                            projection: Projection,
                            axis: int,
                            *,
                            mask_catalogue: bool,
                            mask_sensor: bool) -> np.ndarray[float]:
        return np.min(self.distance_sky(projection, mask_catalogue=mask_catalogue, mask_sensor=mask_sensor),
                      axis=axis, initial=np.pi / 2)

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

    def distance_sky(self, projection: Projection, *, mask_catalogue: bool, mask_sensor: bool):
        return self._compute_distances_sky(
            self.sensor_data.stars.project(projection, masked=mask_sensor),
            self.altaz(masked=mask_catalogue),
        )

    def _compute_distances_sky(self, observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float]:
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

        hash_observed = hash_numpy(observed)
        hash_catalogue = hash_numpy(catalogue)

        if hash_observed == self._hash_observed and hash_catalogue == self._hash_catalogue:
            log.debug(f"Returning cached")
        else:
            self._hash_catalogue = hash_catalogue
            self._hash_observed = hash_observed
            observed = np.expand_dims(observed, 1)
            observed[..., 0] = np.pi / 2 - observed[..., 0]
            catalogue = np.expand_dims(catalogue, 0)
            self._cached_distances = spherical_distance(observed, catalogue)
            log.debug(f"Computing sky distance for {observed.shape} × {catalogue.shape}: {np.sum(self._cached_distances)}")
        return self._cached_distances

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

    @staticmethod
    def find_nearest_value(dist, *, axis):
        """
        Find the nearest dot to star or vice versa

        axis: int
            0 for the nearest star to every dot
            1 for the nearest dot to every star
        """
        return np.min(dist, axis=axis, initial=np.pi)

    @staticmethod
    def find_nearest_index(dist, *, axis):
        """
        Find the index of the nearest dot to star or vice versa

        axis: int
            0 for the nearest star to every dot
            1 for the nearest dot to every star
        """
        if dist.size > 0:
            return np.argmin(dist, axis=axis)
        else:
            return np.empty(shape=(dist.shape[axis], 0), dtype=int)

    def pair(self, projection: Projection) -> Counsellor:
        # Find which star is the nearest for every dot

        errors = self.distance_sky(projection, mask_catalogue=False, mask_sensor=True)
        nearest = Matchmaker.find_nearest_index(errors, axis=1)
        print(nearest)

        # Filter the catalogue by that index

        sensor_data = self.sensor_data.culled_copy()
        return Counsellor(self.location, self.time, self.projection_cls,
                          catalogue=self.catalogue, sensor_data=sensor_data)
