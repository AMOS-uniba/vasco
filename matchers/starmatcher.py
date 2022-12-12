import numpy as np
import scipy as sp
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from utilities import by_azimuth, polar_to_cart, distance
from projections import BorovickaProjection


class StarMatcher():
    """
    Initial star matcher: loads a catalogue (dec/ra) and a set of points (x/y)
    and tries to find the best possible match at time (t) and location (lat/lon)
    """

    def __init__(self, location, time, projection_cls=BorovickaProjection):
        self.projection_cls = projection_cls
        self.stars = None
        self.update(location, time)

    def update(self, location, time):
        self.location = location
        self.time = time
        if self.stars is not None:
            self.update_altaz()

    def update_altaz(self):
        self.altaz = AltAz(location=self.location, obstime=self.time, pressure=75000 * u.pascal, obswl=500 * u.nm)
        altaz = self.stars.transform_to(self.altaz)
        self.catalogue_altaz = np.stack((altaz.alt.degree, altaz.az.degree))

    def load_sensor(self, filename):
        df = pd.read_csv(filename, sep='\t', header=0)
        df['a_cat_rad'] = np.radians(df['acat'])
        df['z_cat_rad'] = df['zcat'] / 90
        df['x_cat'], df['y_cat'] = polar_to_cart(df['z_cat_rad'], df['a_cat_rad'])
        df['a_com_rad'] = np.radians(df['acom'])
        df['z_com_rad'] = df['zcom'] / 90
        df['x_com'], df['y_com'] = polar_to_cart(df['z_com_rad'], df['a_com_rad'])
        df['dx'], df['dy'] = df['x_cat'] - df['x_com'], df['y_cat'] - df['y_com']
        self.data = df
        self.points = self.data[['x_com', 'y_com']].to_numpy()
        self.values = self.data[['dx', 'dy']].to_numpy()
        self.count = len(self.data)

    def load_catalogue(self, filename: str, lmag: float=10):
        self.catalogue = pd.read_csv(filename, sep='\t', header=1)
        self.catalogue = self.catalogue[self.catalogue.vmag < lmag]
        self.stars = SkyCoord(self.catalogue.ra * u.deg, self.catalogue.dec * u.deg)
        self.update_altaz()

    def sensor_to_sky(self, projection):
        return np.stack(projection(self.points[:, 0], self.points[:, 1]), axis=0)

    def mean_error(self, projection) -> float:
        return np.sqrt(np.sum(self.find_nearest_value(self.sensor_to_sky(projection), self.catalogue_altaz)) / (self.count))

    def compute_distance(self, stars, catalogue):
        stars = np.expand_dims(stars, 0)
        catalogue = np.expand_dims(catalogue, 1)
        catalogue = np.radians(catalogue)
        stars[:, :, 0] = np.pi / 2 - stars[:, :, 0]
        print(stars, catalogue)
        return distance(stars, catalogue)

    def find_nearest_value(self, stars, catalogue):
        dist = self.compute_distance(stars, catalogue)
        nearest = np.min(np.square(dist), axis=0)
        return nearest

    def find_nearest_index(self, stars, catalogue):
        dist = self.compute_distance(stars, catalogue)
        nearest = np.argmin(dist, axis=0)
        return nearest

    def func(self, x):
        return self.mean_error(self.projection_cls(*x))

    def minimize(self, x0=(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0), maxiter=30):
        result = sp.optimize.minimize(self.func, x0, method='Nelder-Mead',
            bounds=((None, None), (None, None), (None, None), (None, None), (None, None), (0, None), (None, None), (None, None), (None, None), (None, None), (0, None), (None, None)),
            options=dict(maxiter=maxiter, disp=True),
        )
        return result
