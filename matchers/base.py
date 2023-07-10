import math
import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

from projections import Projection, BorovickaProjection
from photometry import Calibration
from models import Catalogue, SensorData


class Matcher(metaclass=ABCMeta):
    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.projection_cls = projection_cls
        self.location = None
        self.time = None
        self.catalogue = Catalogue()
        self.sensor_data = SensorData()
        self.update(location, time)

    def load_catalogue(self, filename: str):
        del self.catalogue
        self.catalogue = Catalogue()
        self.catalogue.load(filename)

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

    def update(self, location, time):
        self.location = location
        self.time = time

    @abstractmethod
    def position_errors(self, projection: Projection, *, masked: bool) -> np.ndarray:
        """ Find position error for each dot """

    @abstractmethod
    def position_errors_inverse(self, projection: Projection, *, masked: bool) -> np.ndarray:
        """ Find position error for each star """

    def update_position_smoother(self, projection: Projection, *, bandwidth: float = 0.1):
        pass

    def update_magnitude_smoother(self, projection: Projection, calibration: Calibration, *, bandwidth: float = 0.1):
        pass

    @abstractmethod
    def magnitude_errors(self,
                         projection: Projection,
                         calibration: Calibration,
                         *, masked: bool) -> np.ndarray:
        """ Find magnitude error for each dot """

    @staticmethod
    def avg_error(errors) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(errors)) / errors.size)

    @staticmethod
    def max_error(errors) -> float:
        return np.max(errors, initial=0)

    @abstractmethod
    def print_meteor(self, projection: Projection, calibration: Calibration) -> str:
        """ Correct a meteor and return a XML fragment """

    def func(self, x):
        return self.avg_error(self.position_errors(self.projection_cls(*x), masked=True))

    def minimize(self, x0=(0, 0, 0, 0, 0, math.tau / 4, 0, 0, 0, 0, 0, 0), maxiter=30):
        result = sp.optimize.minimize(
            self.func, x0,
            method='Nelder-Mead',
            bounds=(
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None),  # V
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None),  # epsilon
                (None, None),
            ),
            options=dict(maxiter=maxiter, disp=True),
            #callback=lambda x: print(x, np.degrees(self.func(x))),
        )
        return result
