import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

from projections import BorovickaProjection


class Matcher(metaclass=ABCMeta):
    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.projection_cls = projection_cls
        self.location = None
        self.time = None
        self.catalogue = None
        self.sensor_data = None
        self.update(location, time)

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

    @staticmethod
    def avg_error(errors) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(errors)) / errors.size)

    @staticmethod
    def max_error(errors) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.max(errors)

    def func(self, x):
        return self.avg_error(self.errors(self.projection_cls(*x), True))

    def minimize(self, x0=(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0), maxiter=30):
        result = sp.optimize.minimize(
            self.func, x0,
            method='Nelder-Mead',
            bounds=(
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None), # V
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None), # epsilon
                (None, None),
            ),
            options=dict(maxiter=maxiter, disp=True),
            callback=lambda x: print(x),
        )
        return result
