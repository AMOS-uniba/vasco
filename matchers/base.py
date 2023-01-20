import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod

from projections import BorovickaProjection


class Matcher(metaclass=ABCMeta):
    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.projection_cls = projection_cls
        self.update(location, time)

    @property
    @abstractmethod
    def count(self):
        pass

    @abstractmethod
    def mask_catalogue(self, mask):
        pass

    @abstractmethod
    def mask_sensor_data(self, mask):
        pass

    def reset_mask(self):
        self.catalogue.reset_mask()
        self.sensor_data.reset_mask()

    def update(self, location, time):
        self.location = location
        self.time = time

    def avg_error(self, errors) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(errors)) / errors.size)

    def max_error(self, errors) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.max(errors)

    def minimize(self, x0=(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0), maxiter=30):
        result = sp.optimize.minimize(self.func, x0, method='Nelder-Mead',
            bounds=(
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None), #V
                (None, None),
                (None, None),
                (None, None),
                (None, None),
                (0, None), #epsilon
                (None, None),
            ),
            options=dict(maxiter=maxiter, disp=True),
            callback=lambda x: print(x),
        )
        return result
