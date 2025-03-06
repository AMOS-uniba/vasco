import logging
import math
import dotmap
import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional
from pathlib import Path

from astropy.coordinates import EarthLocation
from astropy.time import Time

from amosutils.catalogue import Catalogue
from amosutils.projections import Projection, BorovickaProjection

from photometry import Calibration
from models import SensorData

log = logging.getLogger('vasco')


class Matcher(metaclass=ABCMeta):
    """
    The base class for matching sensor data to the catalogue.
    """

    def __init__(self,
                 location: EarthLocation,
                 time: Time,
                 projection_cls=BorovickaProjection, *,
                 catalogue: Optional[Catalogue] = None,
                 sensor_data: Optional[SensorData] = None):
        self._altaz = None
        self.projection_cls: BorovickaProjection = projection_cls
        self.location: EarthLocation = None
        self.time: Time = None
        self.catalogue = Catalogue() if catalogue is None else catalogue
        self.sensor_data = SensorData() if sensor_data is None else sensor_data
        self.update(location, time)

    def load_catalogue(self, filename: Path):
        del self.catalogue
        self.catalogue = Catalogue(filename)

    @property
    def valid(self) -> bool:
        return self.catalogue is not None and self.sensor_data is not None

    @abstractmethod
    def mask_catalogue(self, mask):
        """ Apply a mask to the catalogue data """

    @abstractmethod
    def mask_sensor_data(self, mask):
        """ Apply a mask to the sensor data """

    @abstractmethod
    def pair(self, projection):
        """ Assign the nearest catalogue star to every dot """

    def reset_mask(self):
        self.catalogue.reset_mask()
        self.sensor_data.reset_mask()

    def update(self, location: EarthLocation, time: Time) -> None:
        """ Update the internal state. """
        self.location = location
        self.time = time

    @abstractmethod
    def position_errors(self, projection: Projection, *, masked: bool) -> np.ndarray:
        """ Find position error for each dot. """

    @abstractmethod
    def position_errors_inverse(self, projection: Projection, *, masked: bool) -> np.ndarray:
        """ Find position error for each star. """

    def update_position_smoother(self, projection: Projection, *, bandwidth: float = 0.1):
        pass

    def update_magnitude_smoother(self, projection: Projection, calibration: Calibration, *, bandwidth: float = 0.1):
        pass

    @abstractmethod
    def magnitude_errors(self,
                         projection: Projection,
                         calibration: Calibration,
                         *,
                         masked: bool) -> np.ndarray:
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

    @abstractmethod
    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        pass

#    @abstractmethod
#    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
#        """ Correct a meteor and return an XML fragment """

    def func(self, x):
        return self.rms_error(self.position_errors(self.projection_cls(*x), masked=True))

    @staticmethod
    def _get_optimization_parameters(x0, mask):
        return x0[~mask]

    def _build_optimization_function(self,
                                     mask: np.ndarray[float]) -> Callable[[np.ndarray[float], ...], float]:
        """
        Split the parameter vector into immutable and variable part depending on mask
        and return a loss function in which only variable parameters are to be optimized
        and immutable ones are treated as constants
        """
        ifixed = np.where(~mask)
        ivariable = np.where(mask)

        def func(x: np.ndarray[float], *args) -> float:
            variable = np.array(x)

            vec = np.zeros(shape=(12,))
            np.put(vec, ivariable, variable)
            np.put(vec, ifixed, np.array(args))

            return self.rms_error(self.position_errors(self.projection_cls(*vec), masked=True))

        return func

    def get_optimization_bounds(self, mask):
        return self.projection_cls.bounds[mask]

    def minimize(self,
                 x0=np.array((0, 0, 0, 0, math.tau / 4, 1, 0, 0, 0, 0, 0, 0)),
                 maxiter=30,
                 *,
                 mask=np.ones(shape=(12,), dtype=bool)):

        self._altaz = self.catalogue.to_altaz(self.location, self.time, masked=True)
        func = self._build_optimization_function(mask)
        args = self._get_optimization_parameters(np.array(x0), mask)

        if np.count_nonzero(mask) == 0:
            log.warning("At least one parameter must be allowed to vary")
            return tuple(x0)

        result = sp.optimize.minimize(
            func,
            x0[mask],
            args,
            method='Nelder-Mead',
            bounds=self.get_optimization_bounds(mask),
            options=dict(maxiter=maxiter, disp=True),
            callback=lambda x: log.debug(x),
        )

        # Restore the full parameter vector from immutable original args and variable optimized args
        vec = np.zeros(shape=(12,))
        np.put(vec, np.where(mask), result.x)
        np.put(vec, np.where(~mask), args)

        return tuple(vec)
