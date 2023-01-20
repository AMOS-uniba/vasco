import numpy as np
import pandas as pd

from .base import Matcher

from utilities import spherical_distance

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

class Counselor(Matcher):
    def __init__(self, location, time, projection_cls, catalogue, sensor_data):
        super().__init__(location, time, projection_cls)
        # a Counselor has fixed pairs: they have to be set on creation
        assert sensor_data.count == catalogue.count
        self.catalogue = catalogue
        self.sensor_data = sensor_data

        print(f"Counselor created with {self.catalogue.count} pairs:")
        print(self.catalogue)
        print(self.sensor_data)

    @property
    def count(self):
        return self.catalogue.count

    def mask_catalogue(self, mask):
        self.catalogue.set_mask(mask)
        self.sensor_data.set_mask(mask)

    def mask_sensor_data(self, mask):
        self.mask_catalogue(mask)

    def compute_distances(self, observed, catalogue):
        """
        Compute distance matrix for observed points projected to the sky and catalogue stars
        observed:   np.ndarray(N, 2)
        catalogue:  np.ndarray(N, 2)

        Returns
        -------
        np.ndarray(N)
        """
        catalogue = np.radians(catalogue)
        observed[..., 0] = np.pi / 2 - observed[..., 0]   # Convert observed altitude to zenith distance
        return spherical_distance(observed, catalogue)

    def compute_vector_errors(self, observed, catalogue):
        """
        Returns
        np.ndarray(N, 2)
        """
        catalogue = np.radians(catalogue)
        observed[..., 0] = np.pi / 2 - observed[..., 0]   # Convert observed altitude to zenith distance
        return spherical_difference(observed, catalogue)

    def errors(self, projection, masked=False):
        return self.compute_distances(
            self.sensor_data.project(projection, masked=masked),
            self.catalogue.to_altaz_deg(self.location, self.time, masked=masked),
        )

    def errors_inverse(self, projection, masked=False):
        return self.errors(projection, masked)

    def func(self, x):
        return self.avg_error(self.errors(self.projection_cls(*x)))

    def pair(self, projection):
        self.catalogue.cull()
        self.sensor_data.cull()
        return self

    def save(self, filename):
        self.df.to_csv(sep='\t', float_format='.6f', index=False, header=['x', 'y', 'dec', 'ra'])
