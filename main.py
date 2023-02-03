#!/usr/bin/env python

import sys
import yaml
import dotmap
import datetime
import zoneinfo
import numpy as np

from astropy import units as u
from astropy.coordinates import EarthLocation

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow

import matplotlib as mpl
from matplotlib import pyplot as plt

from matchers import Matchmaker, Counselor
from projections import BorovickaProjection
from plots import SensorPlot, SkyPlot, ErrorPlot, VectorErrorPlot
from utilities import masked_grid

from main_ui import Ui_MainWindow

from amos import AMOS, Station

mpl.use('Qt5Agg')

COUNT = 100
GRID_RESOLUTION = 31


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.errors = None
        self.location = None
        self.time = None
        self.projection = None
        self.matcher = None

        self.setupUi(self)
        self.param_widgets = [
            (self.dsb_x0, 'x0'), (self.dsb_y0, 'y0'), (self.dsb_a0, 'a0'), (self.dsb_A, 'A'), (self.dsb_F, 'F'),
            (self.dsb_V, 'V'), (self.dsb_S, 'S'), (self.dsb_D, 'D'), (self.dsb_P, 'P'), (self.dsb_Q, 'Q'),
            (self.dsb_eps, 'eps'), (self.dsb_E, 'E')
        ]

        self.settings = dotmap.DotMap(dict(
            resolution=dict(left=-1, bottom=-1, right=1, top=1)
        ))

        self.populateStations()

        plt.style.use('dark_background')
        self.sensorPlot = SensorPlot(self.tab_sensor)
        self.skyPlot = SkyPlot(self.tab_sky)
        self.errorPlot = ErrorPlot(self.tab_errors)
        self.vectorErrorPlot = VectorErrorPlot(self.tab_vectors)

        self.updateProjection()

        self.loadYAML('data/20220531_055655.yaml')
        self.matcher.load_catalogue('catalogue/HYG30.tsv')
        self.importConstants('out2.yaml')

        self.onParametersChanged()

        self.connectSignalSlots()

        self.maskSensor() # temporary
        self.pair() # temporary

    def populateStations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        self.cb_stations.currentIndexChanged.connect(self.selectStation)

    def selectStation(self, index):
        if index == 0:
            station = Station("custom", self.dsb_lat.value(), self.dsb_lon.value(), 0)
        else:
            station = list(AMOS.stations.values())[index - 1]

        self.dsb_lat.setValue(station.latitude)
        self.dsb_lon.setValue(station.longitude)

        self.updateMatcher()
        self.onLocationTimeChanged()

    def connectSignalSlots(self):
        self.ac_load.triggered.connect(self.loadYAMLFile)

        self.dsb_x0.valueChanged.connect(self.onParametersChanged)
        self.dsb_y0.valueChanged.connect(self.onParametersChanged)
        self.dsb_a0.valueChanged.connect(self.onParametersChanged)
        self.dsb_V.valueChanged.connect(self.onParametersChanged)
        self.dsb_S.valueChanged.connect(self.onParametersChanged)
        self.dsb_D.valueChanged.connect(self.onParametersChanged)
        self.dsb_P.valueChanged.connect(self.onParametersChanged)
        self.dsb_Q.valueChanged.connect(self.onParametersChanged)
        self.dsb_A.valueChanged.connect(self.onParametersChanged)
        self.dsb_F.valueChanged.connect(self.onParametersChanged)
        self.dsb_eps.valueChanged.connect(self.onParametersChanged)
        self.dsb_E.valueChanged.connect(self.onParametersChanged)

        self.dt_time.dateTimeChanged.connect(self.updateTime)
        self.dt_time.dateTimeChanged.connect(self.onTimeChanged)
        self.dsb_lat.valueChanged.connect(self.onLocationChanged)
        self.dsb_lon.valueChanged.connect(self.onLocationChanged)

        self.pb_optimize.clicked.connect(self.minimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.exportFile)
        self.pb_import.clicked.connect(self.importFile)

        self.pb_mask_unidentified.clicked.connect(self.maskSensor)
        self.pb_mask_distant.clicked.connect(self.maskCatalogue)
        self.pb_reset.clicked.connect(self.resetValid)
        self.dsb_error_limit.valueChanged.connect(self.onErrorLimitChanged)

        self.dsb_bandwidth.valueChanged.connect(self.onBandwidthChanged)
        self.sb_arrow_scale.valueChanged.connect(self.onArrowScaleChanged)
        self.sb_resolution.valueChanged.connect(self.onResolutionChanged)

        self.cb_show_errors.clicked.connect(self.plotVectorErrors)
        self.cb_show_grid.clicked.connect(self.plotVectorGrid)

        self.tw_charts.currentChanged.connect(self.updatePlots)

    def onTimeChanged(self):
        self.updateTime()
        self.onLocationTimeChanged()

    def onLocationChanged(self):
        self.updateLocation()
        self.onLocationTimeChanged()

    def setLocation(self, lat, lon):
        self.dsb_lat.setValue(lat)
        self.dsb_lon.setValue(lon)

    def updateLocation(self):
        self.location = EarthLocation(self.dsb_lon.value() * u.deg, self.dsb_lat.value() * u.deg)

    def setTime(self, time):
        self.dt_time.setDateTime(time)

    def updateTime(self):
        self.time = self.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss')

    def updateMatcher(self):
        self.matcher.update(self.location, self.time)
        self.matcher.update_smoother(self.projection)

    def updateProjection(self):
        self.projection = BorovickaProjection(*self.get_constants_tuple())

    def onParametersChanged(self):
        self.updateProjection()

        self.skyPlot.invalidate_dots()
        self.skyPlot.invalidate_meteor()
        self.errorPlot.invalidate()
        self.vectorErrorPlot.invalidate_dots()
        self.vectorErrorPlot.invalidate_grid()
        self.vectorErrorPlot.invalidate_meteor()
        self.matcher.update_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())

        self.computeErrors()
        self.updatePlots()

    def onLocationTimeChanged(self):
        self.updateMatcher()
        self.skyPlot.invalidate_stars()
        self.errorPlot.invalidate()
        self.matcher.update_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())
        self.vectorErrorPlot.invalidate_dots()
        self.vectorErrorPlot.invalidate_grid()
        self.vectorErrorPlot.invalidate_meteor()

        self.computeErrors()
        self.updatePlots()

    def onErrorLimitChanged(self):
        self.skyPlot.invalidate_dots()
        self.errorPlot.invalidate()
        self.vectorErrorPlot.invalidate_dots()
        self.updatePlots()

    def onBandwidthChanged(self):
        self.matcher.update_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())
        self.vectorErrorPlot.invalidate_grid()
        self.vectorErrorPlot.invalidate_meteor()
        self.updatePlots()

    def onArrowScaleChanged(self):
        self.vectorErrorPlot.invalidate_dots()
        self.vectorErrorPlot.invalidate_meteor()
        self.updatePlots()

    def onResolutionChanged(self):
        self.vectorErrorPlot.invalidate_grid()
        self.updatePlots()

    def updatePlots(self):
        self.showErrors()
        index = self.tw_charts.currentIndex()
        if index == 0:
            if not self.sensorPlot.valid:
                self.plotSensorData()
        elif index == 1:
            if not self.skyPlot.valid_dots:
                self.plotObservedStars()
            if not self.skyPlot.valid_stars:
                self.plotCatalogueStars()
        elif index == 2:
            if not self.errorPlot.valid:
                self.plotErrors()
        elif index == 3:
            if not self.vectorErrorPlot.valid_dots:
                self.plotVectorErrors()
            if not self.vectorErrorPlot.valid_meteor:
                self.plotVectorMeteor()
            if not self.vectorErrorPlot.valid_grid:
                self.plotVectorGrid()

    def computeErrors(self):
        self.errors = self.matcher.errors(self.projection, True)

    def exportFile(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export constants to file", ".", "YAML files (*.yaml)")
        self.exportConstants(filename)

    def exportConstants(self, filename):
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

    def loadYAMLFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Kvant YAML file", "data", "YAML files (*.yml *.yaml)")
        if filename != '':
            self.loadYAML(filename)

        self.cb_stations.setCurrentIndex(0)
        self.onLocationTimeChanged()
        self.updatePlots()

    def loadYAML(self, file):
        data = dotmap.DotMap(yaml.safe_load(open(file, 'r')))
        self.setLocation(data.Latitude, data.Longitude)
        self.updateLocation()
        self.setTime(datetime.datetime.strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=zoneinfo.ZoneInfo('UTC')))
        self.updateTime()

        self.matcher = Matchmaker(self.location, self.time)
        self.matcher.sensor_data.load(data)

    def importFile(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import constants from file", ".", "YAML files (*.yml *.yaml)")
        self.importConstants(filename)
        self.onParametersChanged()

    def importConstants(self, filename):
        try:
            with open(filename, 'r') as file:
                try:
                    data = dotmap.DotMap(yaml.safe_load(file))
                    for widget, param in self.param_widgets:
                        widget.blockSignals(True)
                        widget.setValue(data.params[param])
                        widget.blockSignals(False)
                    self.updateProjection()
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
        self.w_input.repaint()

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
        self.w_input.repaint()
        self.onParametersChanged()

    def maskSensor(self):
        errors = self.matcher.errors(self.projection, False)
        self.matcher.mask_sensor_data(errors > np.radians(self.dsb_error_limit.value()))
        print(f"Culled the dots to {self.dsb_error_limit.value()}°: {self.matcher.sensor_data.stars.count_valid} are valid")
        self.onParametersChanged()
        self.showCounts()

    def maskCatalogue(self):
        errors = self.matcher.errors_inverse(self.projection, False)
        self.matcher.mask_catalogue(errors > np.radians(self.dsb_distance_limit.value()))
        print(f"Culled the catalogue to {self.dsb_distance_limit.value()}°: {self.matcher.catalogue.count_valid} stars used")
        self.skyPlot.invalidate_stars()

        self.computeErrors()
        self.updatePlots()
        self.showCounts()

    def resetValid(self):
        self.matcher.reset_mask()
        self.onParametersChanged()
        self.showCounts()

    @QtCore.pyqtSlot()
    def showCounts(self):
        if isinstance(self.matcher, Counselor):
            self.lb_mode.setText("paired")
            self.tab_paired.setEnabled(True)
        else:
            self.lb_mode.setText("unpaired")
            self.tab_paired.setEnabled(False)

        self.lb_catalogue_all.setText(f'{self.matcher.catalogue.count}')
        self.lb_catalogue_near.setText(f'{self.matcher.catalogue.count_valid}')
        self.lb_objects_all.setText(f'{self.matcher.sensor_data.stars.count}')
        self.lb_objects_near.setText(f'{self.matcher.sensor_data.stars.count_valid}')

    def showErrors(self):
        avg_error = self.matcher.avg_error(self.errors)
        max_error = self.matcher.max_error(self.errors)
        self.lb_avg_error.setText(f'{np.degrees(avg_error):.6f}°')
        self.lb_max_error.setText(f'{np.degrees(max_error):.6f}°')
        self.lb_total_stars.setText(f'{self.matcher.catalogue.count}')
        outside_limit = self.errors[self.errors > np.radians(self.dsb_error_limit.value())].size
        self.lb_outside_limit.setText(f'{outside_limit}')

    def plotSensorData(self):
        self.sensorPlot.update(self.matcher.sensor_data)

    def plotObservedStars(self):
        self.skyPlot.update_dots(
            self.matcher.sensor_data.stars.project(self.projection, masked=True),
            self.matcher.sensor_data.stars.m,
            self.errors,
            limit=np.radians(self.dsb_error_limit.value())
        )
        self.skyPlot.update_meteor(
            self.matcher.sensor_data.meteor.project(self.projection),
            self.matcher.sensor_data.meteor.m
        )

    def plotCatalogueStars(self):
        self.skyPlot.update_stars(
            self.matcher.catalogue.to_altaz_chart(self.location, self.time, masked=True),
            self.matcher.catalogue.vmag
        )

    def plotErrors(self):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=True)
        self.errorPlot.update(positions, self.matcher.sensor_data.stars.m, self.errors,
                              limit=np.radians(self.dsb_error_limit.value())
        )

    def plotVectorErrors(self):
        # Do nothing if working in unpaired mode
        if isinstance(self.matcher, Counselor):
            if self.cb_show_errors.isChecked():
                print("Plotting vector errors")
                self.vector_tabs.setCurrentIndex(1)
                self.vectorErrorPlot.update_errors(
                    self.matcher.catalogue.altaz(self.location, self.time, masked=True),
                    self.matcher.sensor_data.stars.project(self.projection, masked=True),
                    limit=np.radians(self.dsb_error_limit.value()),
                    scale=1 / self.sb_arrow_scale.value(),
                )
            else:
                self.vectorErrorPlot.clear_errors()
        else:
            self.vector_tabs.setCurrentIndex(0)

    def plotVectorMeteor(self):
        if isinstance(self.matcher, Counselor):
            print("Plotting vector meteors")
            self.vectorErrorPlot.update_meteor(
                self.matcher.sensor_data.meteor.project(self.projection),
                self.matcher.correct_meteor(self.projection),
                self.matcher.sensor_data.meteor.ms(True),
                scale=1 / self.sb_arrow_scale.value(),
            )
        else:
            self.vector_tabs.setCurrentIndex(0)

    def plotVectorGrid(self):
        if isinstance(self.matcher, Counselor):
            self.vector_tabs.setCurrentIndex(1)

            if self.cb_show_grid.isChecked():
                print("Plotting vector grid")
                grid = self.matcher.grid(resolution=self.sb_resolution.value())

                xx, yy = masked_grid(self.sb_resolution.value())
                self.vectorErrorPlot.update_grid(
                    xx, yy, grid[..., 0].ravel(), grid[..., 1].ravel()
                )
            else:
                self.vectorErrorPlot.clear_grid()
        else:
            self.vector_tabs.setCurrentIndex(0)

    def pair(self):
        self.matcher = self.matcher.pair(self.projection)
        self.matcher.update_smoother(self.projection)

        self.skyPlot.invalidate_dots()
        self.skyPlot.invalidate_stars()
        self.errorPlot.invalidate()
        self.vectorErrorPlot.valid_dots = False
        self.vectorErrorPlot.valid_grid = False
        self.vectorErrorPlot.valid_meteor = False
        self.showCounts()
        self.updatePlots()


app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()

app.exec()
