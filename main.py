#!/usr/bin/env python

import sys
import yaml
import numpy as np
import scipy as sp
import pandas as pd

from typing import Tuple, Type, Optional

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

from matchers import StarMatcher
from projections import Projection, EquidistantProjection, BorovickaProjection

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


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        print("Init")
        super().__init__(parent)
        self.setupUi(self)
        self.connectSignalSlots()

        self.populateStations()

        self.setupSensorPlot()
        self.setupSkyPlot()

        self.sensorScatter = self.sensorAxis.scatter([0], [0], s=50, c='red', marker='x')
        self.skyScatter = self.skyAxis.scatter([0], [0], s=50, c='red', marker='x')
        self.starsScatter = self.skyAxis.scatter([0], [0])

        self.w_plots.layout().addWidget(self.sensorCanvas)
        self.w_plots.layout().addWidget(self.skyCanvas)

        self.update_projection()
        self.update_time()
        self.update_location()

        self.matcher = StarMatcher(self.location, self.time)
        self.matcher.load_sensor('data/2016-11-23-084800.tsv')
        self.matcher.load_catalogue('catalogue/HYG30.tsv', lmag=4.5)
        self.update_matcher()

        self.sensorScatter.set_offsets(np.stack((self.matcher.points[:, 0], self.matcher.points[:, 1]), axis=1))
        self.sensorCanvas.draw()

        self.compute_error()
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
        self.plot_stars()

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

        self.dt_time.dateTimeChanged.connect(self.update_time)
        self.dt_time.dateTimeChanged.connect(self.update_matcher)
        self.dt_time.dateTimeChanged.connect(self.plot_stars)

        self.dsb_lat.valueChanged.connect(self.update_location)
        self.dsb_lat.valueChanged.connect(self.update_matcher)
        self.dsb_lon.valueChanged.connect(self.update_location)
        self.dsb_lon.valueChanged.connect(self.update_matcher)
        self.pb_plot.clicked.connect(self.plot_stars)

        self.pb_optimize.clicked.connect(self.minimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.export)

    def update_location(self):
        self.location = EarthLocation(self.dsb_lon.value() * u.deg, self.dsb_lat.value() * u.deg)

    def update_time(self):
        self.time = self.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')

    def update_matcher(self):
        self.matcher.update(self.location, self.time)

    def update_projection(self):
        self.projection = BorovickaProjection(*self.get_tuple())

    def on_parameters_changed(self):
        self.update_projection()
        self.compute_error()
        self.plot()

    def export(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export constants")
        with open(filename, 'w+') as file:
            yaml.dump(dict(
                proj='Borovicka',
                params=dict(
                    x0=self.dsb_x0.value(),
                    y0=self.dsb_y0.value(),
                    a0=self.dsb_a0.value(),
                    F=self.dsb_F.value(),
                    V=self.dsb_V.value(),
                    S=self.dsb_S.value(),
                    D=self.dsb_D.value(),
                    P=self.dsb_P.value(),
                    Q=self.dsb_Q.value(),
                    eps=self.dsb_eps.value(),
                    E=self.dsb_E.value(),
                )
            ), file)

    def get_tuple(self):
        return (self.dsb_x0.value(),
            self.dsb_y0.value(),
            np.radians(self.dsb_a0.value()),
            self.dsb_A.value(),
            np.radians(self.dsb_F.value()),
            self.dsb_V.value(),
            self.dsb_S.value(),
            self.dsb_D.value(),
            self.dsb_P.value(),
            self.dsb_Q.value(),
            np.radians(self.dsb_eps.value()),
            np.radians(self.dsb_E.value()),
        )

    def minimize(self):
        result = self.matcher.minimize(x0=self.get_tuple(), maxiter=self.sb_maxiter.value())
        x0, y0, a0, A, F, V, S, D, P, Q, e, E = tuple(result.x)
        self.dsb_x0.setValue(x0)
        self.dsb_y0.setValue(y0)
        self.dsb_a0.setValue(np.degrees(a0))
        self.dsb_A.setValue(A)
        self.dsb_F.setValue(np.degrees(F))
        self.dsb_V.setValue(V)
        self.dsb_S.setValue(S)
        self.dsb_D.setValue(D)
        self.dsb_P.setValue(P)
        self.dsb_Q.setValue(Q)
        self.dsb_eps.setValue(np.degrees(e))
        self.dsb_E.setValue(np.degrees(E))
        print(result.x, np.degrees(result.fun))

    def plot(self):
        z, a = self.matcher.sensor_to_sky(self.projection)

        self.skyScatter.set_offsets(np.stack((a, np.degrees(z)), axis=1))
        self.skyCanvas.draw()

    def plot_stars(self):
        z, a = self.matcher.catalogue_altaz
        a = np.radians(a)
        z = 90 - z

        s = np.exp(-0.666 * (self.matcher.catalogue['vmag'] - 5))
        self.starsScatter.set_offsets(np.stack((a, z), axis=1))
        self.starsScatter.set_sizes(s)
        self.skyCanvas.draw()

    def compute_error(self):
        error = self.matcher.mean_error(self.projection)
        self.lb_error.setText(f'Mean error: {np.degrees(error):.6f}°')

    def pair(self):
        self.matcher.pair(self.projection, self.location, self.time)



app = QApplication(sys.argv)

window = MainWindow()
window.showMaximized()

app.exec()
