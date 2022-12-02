#!/usr/bin/env python

import sys
import numpy as np
import scipy as sp
import pandas as pd

from typing import Tuple, Type, Optional
from utilities import by_azimuth, polar_to_cart

import astropy
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time

import PyQt6
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from shifters import OpticalAxisShifter, EllipticShifter
from transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer
from projections import Projection, BorovickaProjection

from main_ui import Ui_MainWindow

from amos import AMOS

mpl.use('Qt5Agg')

COUNT = 100


class Fitter():
    def __init__(self):
        pass

    def __call__(self, xy: Tuple[np.ndarray, np.ndarray], za: Tuple[np.ndarray, np.ndarray], cls: Type[Projection], *, params: Optional[dict]=None) -> Projection:
        """
            xy      a 2-tuple of x and y coordinates on the sensor
            za      a 2-tuple of z and a coordinates in the sky catalogue
            cls     a subclass of Projection that is used to transform xy onto za
            Returns an instance of cls with parameters set to values that result in minimal deviation
        """
        return cls(params)


def spherical(x: AltAz, y: AltAz) -> u.Quantity:
    return 2 * np.sin(np.sqrt(np.sin(0.5 * (y.alt - x.alt))**2 + np.cos(x.alt) * np.cos(y.alt) * np.sin(0.5 * (y.az - x.az))**2) * u.rad)


class Vasco():
    """ Virtual All-Sky CorrectOr Plate """
    def __init__(self):
        pass
#        self.argparser = argparse.ArgumentParser("Virtual all-sky corrector plate")
#        self.argparser.add_argument('infile', type=argparse.FileType('r'), help="input file")
#        self.argparser.add_argument('outdir', action=argparser.WriteableDir, help="output directory")
#        self.argparser.add_argument('method', type=str, choices=Corrector.METHODS)
#        self.args = self.argparser.parse_args()
#        self.outdir = Path(self.args.outdir)

    def load(self, filename):
        df = pd.read_csv(filename, sep='\t', header=0, nrows=500)
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

    def load_catalogue(self, filename):
        df = pd.read_csv(filename, sep='\t', header=1)
        self.catalogue = df[df.vmag < 5]
        self.stars = SkyCoord(self.catalogue.ra * u.deg, self.catalogue.dec * u.deg)

    def catalogue_altaz(self, location, time):
        self.altaz = AltAz(location=location, obstime=time, pressure=75000 * u.pascal, obswl=500 * u.nm)
        altaz = self.stars.transform_to(self.altaz)
        return np.stack((altaz.alt.degree, altaz.az.degree))


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        print("Init")
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalSlots()

        self.vasco = Vasco()
        self.vasco.load('data/2016-11-23-084800.tsv')
        self.vasco.load_catalogue('catalogue/HYG30.tsv')
        self.populateStations()

        self.setupSensorPlot()
        self.setupSkyPlot()

        self.sensorScatter = self.sensorAxis.scatter([0], [0])
        self.skyScatter = self.skyAxis.scatter([0], [0])
        self.starsScatter = self.skyAxis.scatter([0], [0])

        self.sensorScatter.set_offsets(np.stack((self.vasco.points[:, 0], self.vasco.points[:, 1]), axis=1))
        self.sensorCanvas.draw()

        self.w_plots.layout().addWidget(self.sensorCanvas)
        self.w_plots.layout().addWidget(self.skyCanvas)

        self.update_projection()

        self.plot()
        self.plot_stars()

    def populateStations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        self.cb_stations.currentIndexChanged.connect(self.selectStation)

    def selectStation(self, index):
        station = list(AMOS.stations.values())[index]
        self.dsb_lat.setValue(station.latitude)
        self.dsb_lon.setValue(station.longitude)

    def setupSensorPlot(self):
        self.sensorFigure = Figure(figsize=(6, 6))
        self.sensorCanvas = FigureCanvasQTAgg(self.sensorFigure)
        self.sensorAxis = self.sensorFigure.add_subplot()
        self.sensorFigure.tight_layout()

        self.sensorAxis.set_xlim([-1, 1])
        self.sensorAxis.set_ylim([-1, 1])
        self.sensorAxis.grid()
        self.sensorAxis.set_aspect('equal')

    def setupSkyPlot(self):
        self.skyFigure = Figure(figsize=(6, 6))
        self.skyCanvas = FigureCanvasQTAgg(self.skyFigure)
        self.skyAxis = self.skyFigure.add_subplot(projection='polar')
        self.skyFigure.tight_layout()

        self.skyAxis.set_ylim([0, 90])
        self.skyAxis.set_theta_offset(3 * np.pi / 2)

    def connectSignalSlots(self):
        print("Connecting signals")
        self.pb_plot.clicked.connect(self.plot)
        self.dsb_x0.valueChanged.connect(self.on_parameters_changed)
        self.dsb_y0.valueChanged.connect(self.on_parameters_changed)
        self.dsb_a0.valueChanged.connect(self.on_parameters_changed)
        self.dsb_V.valueChanged.connect(self.on_parameters_changed)
        self.dsb_S.valueChanged.connect(self.on_parameters_changed)
        self.dsb_D.valueChanged.connect(self.on_parameters_changed)
        self.dsb_P.valueChanged.connect(self.on_parameters_changed)
        self.dsb_Q.valueChanged.connect(self.on_parameters_changed)
        self.dsb_A.valueChanged.connect(self.on_parameters_changed)
        self.dsb_F.valueChanged.connect(self.on_parameters_changed)
        self.dsb_eps.valueChanged.connect(self.on_parameters_changed)
        self.dsb_E.valueChanged.connect(self.on_parameters_changed)

        self.dt_time.dateTimeChanged.connect(self.plot_stars)
        self.pb_plot.clicked.connect(self.plot_stars)
        self.pb_identify.clicked.connect(self.identify)

    def get_location(self):
        return EarthLocation(self.dsb_lon.value() * u.deg, self.dsb_lat.value() * u.deg)

    def on_parameters_changed(self):
        self.update_projection()
        self.plot()

    def update_projection(self):
        self.proj = BorovickaProjection(
            x0=self.dsb_x0.value(),
            y0=self.dsb_y0.value(),
            a0=np.radians(self.dsb_a0.value()),
            V=self.dsb_V.value(),
            S=self.dsb_S.value(),
            D=self.dsb_D.value(),
            P=self.dsb_P.value(),
            Q=self.dsb_Q.value(),
            A=self.dsb_A.value(),
            F=np.radians(self.dsb_F.value()),
            epsilon=np.radians(self.dsb_eps.value()),
            E=np.radians(self.dsb_E.value()),
        )
        print(f"New proj: {self.proj}")

    def plot(self):
        x, y = self.vasco.points[:, 0], self.vasco.points[:, 1]
        z, a = self.proj(x, y)

        self.skyScatter.set_offsets(np.stack((a, np.degrees(z)), axis=1))
        self.skyCanvas.draw()

    def plot_stars(self):
        print("Plotting stars")
        z, a = self.vasco.catalogue_altaz(self.get_location(), self.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss'))
        a = np.radians(a)
        z = 90 - z

        s = np.exp(-0.666 * (self.vasco.catalogue['vmag'] - 5))
        self.starsScatter.set_offsets(np.stack((a, z), axis=1))
        self.starsScatter.set_sizes(s)
        self.skyCanvas.draw()

    def identify(self):
        x, y = self.vasco.points[:, 0], self.vasco.points[:, 1]

        metric = self.find_nearest(
            np.stack(self.proj(x, y), axis=1),
            np.stack(self.vasco.catalogue_altaz(self.get_location(), self.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')), axis=1)
        )
        self.lb_error.setText(f'{metric:.8f}')

    def find_nearest(self, stars, catalogue):
        print(f"Finding nearest stars...")
        stars = np.expand_dims(stars, 0)
        catalogue = np.expand_dims(catalogue, 1)
        catalogue = np.radians(catalogue)
        stars[:, :, 0] = np.pi / 2 - stars[:, :, 0]

        dist = distance(stars, catalogue)
        nearest = np.min(np.square(dist), axis=0)

        return np.sum(nearest)


def distance(x, y):
    return 2 * np.sin(
        np.sqrt(
            np.sin(0.5 * (y[:, :, 0] - x[:, :, 0]))**2 +
            np.cos(x[:, :, 0]) * np.cos(y[:, :, 0]) * np.sin(0.5 * (y[:, :, 1] - x[:, :, 1]))**2.0
        )
    )


app = QApplication(sys.argv)

window = MainWindow()
window.showMaximized()

app.exec()
