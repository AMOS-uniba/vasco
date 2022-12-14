#!/usr/bin/env python

import sys
import yaml
import dotmap
import numpy as np

from typing import Tuple, Type, Optional

from astropy import units as u
from astropy.coordinates import EarthLocation

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow

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
        self.setupErrorPlot()

        self.sensorScatter = self.sensorAxis.scatter([0], [0], s=50, c='red', marker='x')
        self.skyScatter = self.skyAxis.scatter([0], [0], s=50, c='red', marker='x')
        self.starsScatter = self.skyAxis.scatter([0], [0], marker='o', c='white')
        self.errorScatter = self.errorAxis.scatter([0], [0], marker='x', c='cyan')
        self.skyQuiver = self.skyAxis.quiver([0], [0], [0], [0])

        self.tab_sensor.layout().addWidget(self.sensorCanvas)
        self.tab_sky.layout().addWidget(self.skyCanvas)
        self.tab_errors.layout().addWidget(self.errorCanvas)

        self.update_projection()
        self.update_time()
        self.update_location()

        self.matcher = StarMatcher(self.location, self.time)
        self.matcher.load_sensor('data/2016-11-23-084800.tsv')
        self.matcher.load_catalogue('catalogue/HYG30.tsv')
        self.matcher.catalogue.filter(5)
        self.update_matcher()

        self.sensorScatter.set_offsets(self.matcher.sensor_data.points)
        self.sensorCanvas.draw()

        self.compute_error()
        self.plot_observed_stars()
        self.plot_catalogue_stars()
        self.plot_errors()

    def populateStations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        self.cb_stations.currentIndexChanged.connect(self.selectStation)

    def selectStation(self, index):
        station = list(AMOS.stations.values())[index]
        self.dsb_lat.setValue(station.latitude)
        self.dsb_lon.setValue(station.longitude)
        self.update_matcher()
        self.plot_catalogue_stars()
        self.compute_error()

    def setupSensorPlot(self):
        plt.style.use('dark_background')
        self.sensorFigure = Figure(figsize=(6, 6))
        self.sensorCanvas = FigureCanvasQTAgg(self.sensorFigure)
        self.sensorAxis = self.sensorFigure.add_subplot()
        self.sensorFigure.tight_layout()

        self.sensorAxis.set_xlim([-1, 1])
        self.sensorAxis.set_ylim([-1, 1])
        self.sensorAxis.grid(color='white', alpha=0.3)
        self.sensorAxis.set_aspect('equal')

    def setupSkyPlot(self):
        self.skyFigure = Figure(figsize=(6, 6))
        self.skyCanvas = FigureCanvasQTAgg(self.skyFigure)
        self.skyAxis = self.skyFigure.add_subplot(projection='polar')
        self.skyFigure.tight_layout()

        self.skyAxis.set_xlim([0, 2 * np.pi])
        self.skyAxis.set_ylim([0, 90])
        self.skyAxis.set_rlabel_position(0)
        self.skyAxis.set_rticks([15, 30, 45, 60, 75])
        self.skyAxis.yaxis.set_major_formatter('{x}°')
        self.skyAxis.grid(color='white', alpha=0.3)
        self.skyAxis.set_theta_offset(3 * np.pi / 2)

    def setupErrorPlot(self):
        self.errorFigure = Figure(figsize=(8, 6))
        self.errorCanvas = FigureCanvasQTAgg(self.errorFigure)
        self.errorAxis = self.errorFigure.add_subplot()
        self.errorFigure.tight_layout()

        self.errorAxis.set_xlim([0, 90])
        self.errorAxis.set_ylim([0, None])

    def connectSignalSlots(self):
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
        self.dt_time.dateTimeChanged.connect(self.plot_catalogue_stars)

        self.dsb_lat.valueChanged.connect(self.update_location)
        self.dsb_lat.valueChanged.connect(self.update_matcher)
        self.dsb_lon.valueChanged.connect(self.update_location)
        self.dsb_lon.valueChanged.connect(self.update_matcher)

        self.pb_optimize.clicked.connect(self.minimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.export_file)
        self.pb_import.clicked.connect(self.import_file)

    def update_location(self):
        self.location = EarthLocation(self.dsb_lon.value() * u.deg, self.dsb_lat.value() * u.deg)

    def update_time(self):
        self.time = self.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')

    def update_matcher(self):
        self.matcher.update(self.location, self.time)

    def update_projection(self):
        self.projection = BorovickaProjection(*self.get_constants_tuple())

    def on_parameters_changed(self):
        self.update_projection()
        self.compute_error()
        self.plot_observed_stars()
        self.plot_errors()

    def export_file(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export constants to file", ".", "YAML files (*.yaml)")
        self.export_constants(filename)

    def export_constants(self, filename):
        try:
            with open(filename, 'w+') as file:
                yaml.dump(dict(
                    proj='Borovicka',
                    params=dict(
                        x0=self.dsb_x0.value(),
                        y0=self.dsb_y0.value(),
                        a0=self.dsb_a0.value(),
                        A=self.dsb_A.value(),
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
        except FileNotFoundError as exc:
            print(f"Could not export constants: {exc}")

    def import_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import constants from file", ".", "YAML files (*.yml *.yaml)")
        self.import_constants(filename)

    def import_constants(self, filename):
        try:
            with open(filename, 'r') as file:
                try:
                    data = yaml.safe_load(file)
                    data = dotmap.DotMap(data)
                    self.dsb_x0.setValue(data.params.x0)
                    self.dsb_y0.setValue(data.params.y0)
                    self.dsb_a0.setValue(data.params.a0)
                    self.dsb_A.setValue(data.params.A)
                    self.dsb_F.setValue(data.params.F)
                    self.dsb_V.setValue(data.params.V)
                    self.dsb_S.setValue(data.params.S)
                    self.dsb_D.setValue(data.params.D)
                    self.dsb_P.setValue(data.params.P)
                    self.dsb_Q.setValue(data.params.Q)
                    self.dsb_eps.setValue(data.params.eps)
                    self.dsb_E.setValue(data.params.E)
                except yaml.YAMLError as exc:
                    print(f"Could not open file {filename}")
        except FileNotFoundError as exc:
            print(f"Could not import constants: {exc}")

    def get_constants_tuple(self):
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
        self.w_input.setEnabled(False)
        self.w_input.repaint();

        result = self.matcher.minimize(
        #    location=self.location,
        #    time=self.time,
            x0=self.get_constants_tuple(),
            maxiter=self.sb_maxiter.value()
        )

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

        self.w_input.setEnabled(True)
        self.w_input.repaint();
        self.plot_errors()

    def plot_observed_stars(self):
        z, a = self.matcher.sensor_data.project(self.projection)
        self.skyScatter.set_offsets(np.stack((a, np.degrees(z)), axis=1))
        self.skyCanvas.draw()

    def plot_catalogue_stars(self):
        z, a = self.matcher.catalogue.to_altaz(self.location, self.time)
        a = np.radians(a)
        offsets = np.stack((a, 90 - z), axis=1)
        self.starsScatter.set_offsets(offsets)

        s = np.exp(-0.666 * (self.matcher.catalogue.vmag - 5))
        self.starsScatter.set_sizes(s)

        self.skyCanvas.draw()

    def plot_quiver(self):
        return
        z, a = self.matcher.sensor_data.project(self.projection)
        offsets = np.stack((a, np.degrees(z)), axis=1)
        self.skyQuiver.set_offsets(offsets)

        zz, aa = self.matcher.catalogue_altaz
        aa = np.radians(aa)
        offsets = np.stack((aa, 90 - zz), axis=1)
        self.skyQuiver.set_UVC(zz - z, aa - a)

        self.skyCanvas.draw()

    def plot_errors(self):
        alt = np.degrees(self.matcher.sensor_data.project(self.projection)[0, :])
        err = np.degrees(self.matcher.errors(self.projection))
        self.errorScatter.set_offsets(np.stack((alt, err), axis=1))
        self.errorAxis.set_ylim([0, 0.2])
        self.errorCanvas.draw()

    def compute_error(self):
        mean_error = self.matcher.mean_error(self.projection)
        self.lb_avg_error.setText(f'Mean error: {np.degrees(mean_error):.6f}°')
        max_error = self.matcher.max_error(self.projection)
        self.lb_max_error.setText(f'Max error: {np.degrees(max_error):.6f}°')

    def pair(self):
        self.matcher.pair(self.projection)
        self.plot_catalogue_stars()
        self.plot_errors()
        self.plot_quiver()



app = QApplication(sys.argv)

window = MainWindow()
window.showMaximized()

app.exec()
