import logging
import math
import dotmap
import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
from pathlib import Path

from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time

from amosutils.catalogue import Catalogue
from amosutils.projections import Projection, BorovickaProjection

from photometry import Calibration
from models import SensorData
from utilities import hash_numpy, spherical_distance

log = logging.getLogger('vasco')


class Matcher(metaclass=ABCMeta):
    """
    The base class for matching sensor data to the catalogue.
    """

    def __init__(self,
                 location: EarthLocation,
                 time: Time,
                 projection_cls: type[Projection] = BorovickaProjection,
                 *,
                 catalogue: Optional[Catalogue] = None,
                 sensor_data: Optional[SensorData] = None):
        self._altaz: Optional[AltAz] = None
        self._altaz_numpy: Optional[np.ndarray] = None
        self._altaz_masked: Optional[np.ndarray] = None

        self._cached_distances = None
        self._hash_observed = None
        self._hash_catalogue = None

        self.projection_cls: type[Projection] = projection_cls
        self.location: Optional[EarthLocation] = None
        self.time: Optional[Time] = None
        self.catalogue = Catalogue() if catalogue is None else catalogue
        self.sensor_data = SensorData() if sensor_data is None else sensor_data
        self.update_location_time(location, time)

        log.debug(f"Created a Matcher ({self.projection_cls.__qualname__})")

    def load_catalogue(self, filename: Path):
        del self.catalogue
        self.catalogue = Catalogue(filename)
        log.info(f"Loaded a catalogue from {filename}: {self.catalogue.count} stars")

    @property
    def valid(self) -> bool:
        return self.catalogue is not None and self.sensor_data is not None

    @abstractmethod
    def mask_catalogue(self, mask):
        """ Apply a mask to the catalogue data """

    @abstractmethod
    def mask_sensor_data(self, mask):
        """ Apply a mask to the sensor data """

    def reset_mask(self):
        self.catalogue.mask = None
        self.sensor_data.stars.mask = None

    def update_location_time(self, location, time):
        """
        Update internal location and time and request the catalogue to recompute the stars' positions.
        """
        self.location = location
        self.time = time

        if self.catalogue.populated:
            self.invalidate_altaz()

    def invalidate_altaz(self):
        log.debug("Invalidating the cached altaz")
        self._altaz = None

    def altaz(self, *, masked: bool):
        """
        Return the current masked catalogue altaz as a numpy array
        """
        if self._altaz is None:
            log.debug(f"Requesting catalogue for {self.location}, {self.time}, {masked}")
            self._altaz = self.catalogue.altaz(self.location, self.time, masked=False)
            self._altaz_masked = self._altaz[self.catalogue.mask]
        else:
            log.debug(f"Requesting catalogue for {self.location}, {self.time}, {masked} (using cached)")

        converted = np.array([self._altaz.alt.radian, self._altaz.az.radian], dtype=float).T
        if masked:
            return converted[self.catalogue.mask]
        else:
            return converted

    def vmag(self, *, masked: bool):
        """
        Return the current masked catalogue vmag as a numpy array
        """
        log.debug("Requesting a new magnitude catalogue")
        vmag = self.catalogue.vmag(self.location, self.time, masked=masked)
        return vmag

    def _compute_distances_sky_cart(self, observed: np.ndarray[float], catalogue: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compute distances in the sky of `observed` and `catalogue` objects
        from the full Cartesian product

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
            observed[..., 0] = math.tau / 4 - observed[..., 0]
            catalogue = np.expand_dims(catalogue, 0)
            self._cached_distances = spherical_distance(observed, catalogue)
            log.debug(f"Computing sky distance for {observed.shape} × {catalogue.shape}: {np.sum(self._cached_distances)}")
        return self._cached_distances


    def distance_sky(self, projection: Projection, *, mask_catalogue: bool, mask_sensor: bool):
        return self._compute_distances_sky_cart(
            self.sensor_data.stars.project(projection, masked=mask_sensor),
            self.altaz(masked=mask_catalogue),
        )

    def position_errors_sky(self,
                            projection: Projection,
                            axis: int,
                            *,
                            mask_catalogue: bool,
                            mask_sensor: bool) -> np.ndarray[float]:
        return np.min(self.distance_sky(projection, mask_catalogue=mask_catalogue, mask_sensor=mask_sensor),
                      axis=axis, initial=np.pi / 2)

    def update_position_smoother(self, projection: Projection, *, bandwidth: float = 0.1):
        pass

    def update_magnitude_smoother(self, projection: Projection, calibration: Calibration, *, bandwidth: float = 0.1):
        pass

    @abstractmethod
    def magnitude_errors_sky(self,
                             projection: Projection,
                             calibration: Calibration,
                             axis: int,
                             *,
                             mask_catalogue: bool,
                             mask_sensor: bool) -> np.ndarray:
        """ Find magnitude error for each dot """

    @staticmethod
    def rms_error(errors: np.ndarray[float]) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(errors)) / errors.size)

    @staticmethod
    def max_error(errors: np.ndarray[float]) -> float:
        return np.max(errors, initial=0)

    @staticmethod
    def find_nearest_value(dist, *, axis: int) -> np.ndarray:
        """
        Find the nearest dot to star or vice versa

        axis: int
            0 for the nearest star to every dot
            1 for the nearest dot to every star
        """
        return np.min(dist, axis=axis, initial=np.pi)

    @staticmethod
    def find_nearest_index(dist, *, axis: int) -> np.ndarray:
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

    @abstractmethod
    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        pass

#    @abstractmethod
#    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
#        """ Correct a meteor and return an XML fragment """

    def func(self, x):
        return self.rms_error(self.position_errors_sky(self.projection_cls(*x), masked=True))

    @staticmethod
    def _get_optimization_parameters(x0, mask):
        return x0[~mask]

    def _build_optimization_function(self,
                                     mask: np.ndarray[float]) -> Callable[[np.ndarray[float], ...], float]:
        """
        Split the parameter vector into immutable and variable part depending on mask.
        Return a loss function in which only variable parameters are to be optimized
        and immutable ones are treated as constants
        """
        ifixed = np.where(~mask)
        ivariable = np.where(mask)

        def func(x: np.ndarray[float], *args) -> float:
            variable = np.array(x)

            vec = np.zeros(shape=(12,))
            np.put(vec, ivariable, variable)
            np.put(vec, ifixed, np.array(args))

            return self.rms_error(self.position_errors_sky(self.projection_cls(*vec), axis=1,
                                                           mask_catalogue=True, mask_sensor=True))

        return func

    def get_optimization_bounds(self, mask):
        return self.projection_cls.bounds[mask]

    def minimize(self,
                 x0=np.array((0, 0, 0, 0, math.tau / 4, 1, 0, 0, 0, 0, 0, 0)),
                 maxiter=30,
                 *,
                 mask=np.ones(shape=(12,), dtype=bool),
                 callback: Callable = lambda x: log.debug(x)):
        func = self._build_optimization_function(mask)
        args = self._get_optimization_parameters(np.array(x0), mask)

        if np.count_nonzero(mask) == 0:
            log.warning("At least one parameter must be allowed to vary")
            return tuple(x0)

        # Do the actual optimization over the selected values
        result = sp.optimize.minimize(
            func,
            x0[mask],
            args,
            method='Nelder-Mead',
            bounds=self.get_optimization_bounds(mask),
            options=dict(maxiter=maxiter, disp=True),
            callback=callback,
        )

        # Restore the full parameter vector from immutable original args and variable optimized args
        vec = np.zeros(shape=(12,))
        np.put(vec, np.where(mask), result.x)
        np.put(vec, np.where(~mask), args)

        return tuple(vec)
