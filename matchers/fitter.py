import numpy as np

from .base import Comparator

#class Fitter():
#    def __init__(self):
#        pass
#
#    def __call__(self, xy: Tuple[np.ndarray, np.ndarray], za: Tuple[np.ndarray, np.ndarray], cls: Type[Projection], *, params: Optional[dict]=None) -> Projection:
#        """
#            xy      a 2-tuple of x and y coordinates on the sensor
#            za      a 2-tuple of z and a coordinates in the sky catalogue
#            cls     a subclass of Projection that is used to transform xy onto za
#            Returns an instance of cls with parameters set to values that result in minimal deviation
#        """
#        return cls(params)

class Fitter(Comparator):
    def __init__(self, sensor, catalogue):
        self.xy = sensor
        self.za = catalogue
        print(f"Fitter created with {sensor} dots and {catalogue} stars")

    def compute_distances(self, stars, catalogue):
        catalogue = np.radians(catalogue)
        stars[:, 0] = np.pi / 2 - stars[:, 0]
        return distance(stars, catalogue)

    def func(self, x, *args):
        projection = BorovickaProjection(*x)
        return self.calculate_error(BorovickaProjection, *args)
