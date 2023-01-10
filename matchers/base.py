import numpy as np
import scipy as sp

from abc import ABCMeta, abstractmethod


class Comparator(metaclass=ABCMeta):
    @abstractmethod
    def minimize(self):
        pass

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
        )
        return result
