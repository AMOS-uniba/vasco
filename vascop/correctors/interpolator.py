import numpy as np
from scipy.interpolate import griddata
from .base import BaseCorrector


class Interpolator(BaseCorrector):
    def __init__(self, points, values, *, method='cubic'):
        super().__init__(points, values)
        self.method = method

    def __call__(self, nodes):
        return griddata(self.points, self.values, nodes, method=self.method, fill_value=np.nan)
