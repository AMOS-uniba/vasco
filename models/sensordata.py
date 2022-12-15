import dotmap
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from utilities import polar_to_cart


class SensorData():
    """ A set of stars in xy format """

    def __init__(self, filename):
        self.rect = dotmap.DotMap(left=-1, right=1, bottom=-1, top=1)

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

    def load(self, data):
        w, h = tuple(map(int, data.Resolution.split('x')))
        self.rect = dotmap.DotMap(dict(left=0, top=0, right=w, bottom=h))

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    def project(self, projection):
        return np.stack(projection(self.x, self.y), axis=0)

