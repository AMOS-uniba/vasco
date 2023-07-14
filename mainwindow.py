import logging

import yaml
import pytz
import datetime
import numpy as np
import dotmap

from PyQt6 import QtCore
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt6.QtCore import QDateTime, Qt

from astropy import units as u
from astropy.coordinates import EarthLocation
from pathlib import Path
import matplotlib as mpl

from matchers import Matchmaker, Counselor
from projections import BorovickaProjection
from plotting import MainWindowPlots
from models import SensorData

import colour as c
from amos import AMOS, Station

mpl.use('Qt5Agg')

log = logging.getLogger('root')

VERSION = "0.7.0"
DATE = "2023-07-13"


class MainWindow(MainWindowPlots):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.populateStations()
        self.updateProjection()

        self.tw_meteor.setColumnWidth(0, 30)

        self.connectSignalSlots()
        self.updateLocation()
        self.updateTime()

        self.resetMatcher()
        self.matcher.load_catalogue('catalogues/HYG30.tsv')
        self._loadSighting('data/20220531_055655.yaml')
        self._importProjectionConstants('calibrations/DRGRmod2.yaml')
        self.showCounts()
        self.onProjectionParametersChanged()
        self.onScalingChanged()

    def connectSignalSlots(self):
        self.ac_load_sighting.triggered.connect(self.loadSighting)
        self.ac_load_catalogue.triggered.connect(self.loadCatalogue)
        self.ac_export_meteor.triggered.connect(self.exportCorrectedMeteor)
        self.ac_mask_unmatched.triggered.connect(self.maskSensor)
        self.ac_create_pairing.triggered.connect(self.pair)
        self.ac_load_constants.triggered.connect(self.importProjectionConstants)
        self.ac_save_constants.triggered.connect(self.exportProjectionConstants)
        self.ac_optimize.triggered.connect(self.optimize)
        self.ac_about.triggered.connect(self.displayAbout)

        for widget in self.param_widgets.values():
            widget.dsb_value.valueChanged.connect(self.onProjectionParametersChanged)

        self.pw_x0.setup(title="H shift", symbol="x<sub>0</sub>", unit="mm", minimum=-5, maximum=5, step=0.001)
        self.pw_y0.setup(title="V shift", symbol="y<sub>0</sub>", unit="mm", minimum=-5, maximum=5, step=0.001)
        self.pw_a0.setup(title="rotation", symbol="a<sub>0</sub>", unit="°", minimum=0, maximum=359.999999, step=0.2,
                         inner_function=np.radians, input_function=np.degrees)

        self.pw_A.setup(title="amplitude", symbol="A", unit="", minimum=-1, maximum=1, step=0.001)
        self.pw_F.setup(title="phase", symbol="F", unit="°", minimum=0, maximum=359.999999, step=1,
                        inner_function=np.radians, input_function=np.degrees)

        self.pw_V.setup(title="linear", symbol="&V", unit="rad/mm", minimum=0.001, maximum=1, step=0.001)
        self.pw_S.setup(title="exp coef", symbol="&S", unit="rad/mm", minimum=-5, maximum=5, step=0.001)
        self.pw_D.setup(title="exp exp", symbol="&D", unit="mm<sup>-1</sup>", minimum=-5, maximum=5, step=0.001)
        self.pw_P.setup(title="biexp coef", symbol="&P", unit="rad/mm", minimum=-5, maximum=5, step=0.001)
        self.pw_Q.setup(title="biexp exp", symbol="&Q", unit="mm<sup>-2</sup>", minimum=-5, maximum=5, step=0.001)

        self.pw_epsilon.setup(title="zenith angle", symbol="ε", unit="°", minimum=0, maximum=90, step=0.1,
                        inner_function=np.radians, input_function=np.degrees)
        self.pw_E.setup(title="azimuth", symbol="E", unit="°", minimum=0, maximum=359.999999, step=1,
                        inner_function=np.radians, input_function=np.degrees)

        self.dt_time.dateTimeChanged.connect(self.updateTime)
        self.dt_time.dateTimeChanged.connect(self.onTimeChanged)

        self.dsb_lat.valueChanged.connect(self.onLocationChanged)
        self.dsb_lon.valueChanged.connect(self.onLocationChanged)
        self.dsb_xs.valueChanged.connect(self.onScalingChanged)
        self.dsb_ys.valueChanged.connect(self.onScalingChanged)

        self.pb_optimize.clicked.connect(self.optimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.exportProjectionConstants)
        self.pb_import.clicked.connect(self.importProjectionConstants)

        self.pb_mask_unidentified.clicked.connect(self.maskSensor)
        self.pb_mask_distant.clicked.connect(self.maskCatalogueDistant)
        self.pb_mask_faint.clicked.connect(self.maskCatalogueFaint)
        self.pb_reset.clicked.connect(self.resetValid)
        self.dsb_error_limit.valueChanged.connect(self.onErrorLimitChanged)

        self.hs_bandwidth.actionTriggered.connect(self.onBandwidthSettingChanged)
        self.hs_bandwidth.sliderMoved.connect(self.onBandwidthSettingChanged)
        self.hs_bandwidth.actionTriggered.connect(self.onBandwidthChanged)
        self.hs_bandwidth.sliderReleased.connect(self.onBandwidthChanged)
        self.sb_arrow_scale.valueChanged.connect(self.onArrowScaleChanged)
        self.sb_resolution.valueChanged.connect(self.onResolutionChanged)

        self.cb_show_errors.clicked.connect(self.plotPositionCorrectionErrors)
        self.cb_show_grid.clicked.connect(self.plotPositionCorrectionGrid)
        self.cb_interpolation.currentIndexChanged.connect(self.plotMagnitudeCorrectionGrid)

        self.tw_charts.currentChanged.connect(self.updatePlots)

    def populateStations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        self.cb_stations.currentIndexChanged.connect(self.selectStation)

    def selectStation(self, index):
        if index == 0:
            station = Station(0, "c", "custom", self.dsb_lat.value(), self.dsb_lon.value(), self.dsb_alt.value())
        else:
            station = list(AMOS.stations.values())[index - 1]

        self.dsb_lat.setValue(station.latitude)
        self.dsb_lon.setValue(station.longitude)
        self.dsb_alt.setValue(station.altitude)

        self.updateMatcher()
        self.onLocationTimeChanged()

    def onTimeChanged(self):
        self.updateTime()
        self.onLocationTimeChanged()

    def onLocationChanged(self):
        self.updateLocation()
        self.onLocationTimeChanged()

    def setLocation(self, lat, lon, alt):
        self.dsb_lat.setValue(lat)
        self.dsb_lon.setValue(lon)
        self.dsb_alt.setValue(alt)

    def updateLocation(self):
        self.location = EarthLocation(
            self.dsb_lon.value() * u.deg,
            self.dsb_lat.value() * u.deg,
            self.dsb_alt.value() * u.m,
        )

    def setTime(self, time):
        self.dt_time.setDateTime(QDateTime(time.date(), time.time(), Qt.TimeSpec.UTC))

    def updateTime(self):
        self.time = self.dt_time.dateTime().toPyDateTime()

    def onScalingChanged(self):
        self.matcher.sensor_data.set_shifter_scales(
            self.dsb_xs.value(),
            self.dsb_ys.value()
        )
        self.onProjectionParametersChanged()

    def updateMatcher(self):
        self.matcher.update(self.location, self.time)
        self.matcher.update_position_smoother(self.projection)

    def updateProjection(self):
        self.projection = BorovickaProjection(*self.getProjectionParameters())

    def onProjectionParametersChanged(self):
        log.info(f"Parameters changed: {self.getProjectionParameters()}")
        self.updateProjection()

        self.positionSkyPlot.invalidate_dots()
        self.positionSkyPlot.invalidate_meteor()
        self.magnitudeSkyPlot.invalidate_dots()
        self.magnitudeSkyPlot.invalidate_meteor()
        self.positionErrorPlot.invalidate()
        self.magnitudeErrorPlot.invalidate()
        self.positionCorrectionPlot.invalidate()
        self.magnitudeCorrectionPlot.invalidate()
        self.matcher.update_position_smoother(self.projection, bandwidth=self.bandwidth())

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()

    def onLocationTimeChanged(self):
        self.updateMatcher()
        self.positionSkyPlot.invalidate_stars()
        self.positionSkyPlot.invalidate_dots()
        self.magnitudeSkyPlot.invalidate_stars()
        self.magnitudeSkyPlot.invalidate_dots()
        self.positionErrorPlot.invalidate()
        self.magnitudeErrorPlot.invalidate()

        bandwidth = self.bandwidth()
        self.matcher.update_position_smoother(self.projection, bandwidth=bandwidth)
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=bandwidth)
        self.positionCorrectionPlot.invalidate()
        self.magnitudeCorrectionPlot.invalidate()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()

    def onErrorLimitChanged(self):
        self.positionSkyPlot.invalidate_dots()
        self.positionErrorPlot.invalidate()
        self.positionCorrectionPlot.invalidate_dots()
        self.updatePlots()

    def onBandwidthSettingChanged(self):
        bandwidth = self.bandwidth()
        self.lb_bandwidth.setText(f"{bandwidth:.03f}")

    def onBandwidthChanged(self, action=0):
        if action == 7: # do not do anything if the user did not drop the slider yet
            return

        bandwidth = self.bandwidth()
        self.matcher.update_position_smoother(self.projection, bandwidth=bandwidth)
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=bandwidth)
        self.positionCorrectionPlot.invalidate_grid()
        self.positionCorrectionPlot.invalidate_meteor()
        self.magnitudeCorrectionPlot.invalidate_grid()
        self.magnitudeCorrectionPlot.invalidate_meteor()
        self.updatePlots()

    def onArrowScaleChanged(self):
        self.positionCorrectionPlot.invalidate_dots()
        self.positionCorrectionPlot.invalidate_meteor()
        self.updatePlots()

    def onResolutionChanged(self):
        self.positionCorrectionPlot.invalidate_grid()
        self.magnitudeCorrectionPlot.invalidate_grid()
        self.updatePlots()

    def bandwidth(self):
        return 10**(-self.hs_bandwidth.value() / 100)

    def resetMatcher(self):
        self.matcher = Matchmaker(self.location, self.time)

    def computePositionErrors(self):
        self.position_errors = self.matcher.position_errors(self.projection, masked=True)

    def computeMagnitudeErrors(self):
        self.magnitude_errors = self.matcher.magnitude_errors(self.projection, self.calibration, masked=True)

    def loadCatalogue(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load catalogue file", "catalogues",
                                                  "Tab-separated values (*.tsv)")
        if filename == '':
            log.warn("No file provided, loading aborted")
        else:
            if self.paired:
                self.resetMatcher()
            self.matcher.load_catalogue(filename)
            self.positionSkyPlot.invalidate_stars()
            self.magnitudeSkyPlot.invalidate_stars()
            self.onProjectionParametersChanged()

    def exportProjectionConstants(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export constants to file", "calibrations",
                                                  "YAML files (*.yaml)")
        if filename is not None and filename != '':
            self._exportProjectionConstants(filename)

    def _exportProjectionConstants(self, filename):
        try:
            with open(filename, 'w+') as file:
                yaml.dump(dict(
                    proj='Borovička',
                    params={ param: widget.inner_value() for param, widget in self.param_widgets.items() },
                ), file)
        except FileNotFoundError as exc:
            log.error(f"Could not export constants: {exc}")

    def loadSighting(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Kvant YAML file", "data",
                                                  "YAML files (*.yml *.yaml)")
        if filename == '':
            log.warn("No file provided, loading aborted")
        else:
            self._loadSighting(filename)

        if (station := AMOS.stations.get(self.matcher.sensor_data.station, None)) is not None:
            log.debug(f"Station {station.code} found, loading properties from AMOS database")
            self.cb_stations.setCurrentIndex(station.id)
            if (path := Path(f'./calibrations/{station.code}.yaml')).exists():
                self._importProjectionConstants(path)
                log.debug(f"Calibration file {path} found, loading projection constants")
            else:
                log.debug(f"Calibration file {path} not found, skipping")
        else:
            log.debug(f"Station not found, marking as custom coordinates")
            self.cb_stations.setCurrentIndex(0)

        self.onLocationTimeChanged()
        self.onProjectionParametersChanged()
        self.sensorPlot.invalidate()
        self.updatePlots()

    def _loadSighting(self, file):
        data = dotmap.DotMap(yaml.safe_load(open(file, 'r')), _dynamic=False)
        self.setLocation(data.Latitude, data.Longitude, data.Altitude)
        self.updateLocation()
        self.setTime(pytz.UTC.localize(datetime.datetime.strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f")))
        self.updateTime()
        self.sensorPlot.invalidate()

        if self.paired:
            self.resetMatcher()
        self.matcher.sensor_data = SensorData.load(data)

    def importProjectionConstants(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import constants from file", "calibrations",
                                                  "YAML files (*.yml *.yaml)")
        self._importProjectionConstants(filename)
        self.onProjectionParametersChanged()

    def _importProjectionConstants(self, filename):
        try:
            with open(filename, 'r') as file:
                try:
                    data = dotmap.DotMap(yaml.safe_load(file), _dynamic=False)
                    self.blockParameterSignals(True)
                    for param, widget in self.param_widgets.items():
                        widget.set_value(data.params[param])
                    self.blockParameterSignals(False)

                    self.updateProjection()
                except yaml.YAMLError as exc:
                    log.error(f"Could not parse file {filename} as YAML: {exc}")
        except FileNotFoundError as exc:
            log.error(f"Could not import constants: {exc}")

    def blockParameterSignals(self, block):
        for widget in self.param_widgets.values():
            widget.dsb_value.blockSignals(block)

    def getProjectionParameters(self):
        return np.array([widget.inner_value() for widget in self.param_widgets.values()], dtype=float)

    def optimize(self):
        self.w_input.setEnabled(False)
        self.w_input.repaint()

        result = self.matcher.minimize(
            x0=self.getProjectionParameters(),
            maxiter=self.sb_maxiter.value(),
            mask=np.array([widget.is_checked() for widget in self.param_widgets.values()], dtype=bool)
        )

        self.blockParameterSignals(True)
        for value, widget in zip(result, self.param_widgets.values()):
            widget.set_from_gui(value)
        self.blockParameterSignals(False)

        self.w_input.setEnabled(True)
        self.w_input.repaint()
        self.onProjectionParametersChanged()

    def updateMeteorTable(self):
        if isinstance(self.matcher, Counselor):
            data = self.matcher.correct_meteor(self.projection, self.calibration)

            self.tw_meteor.setRowCount(data.count)
            for i in range(0, data.count):
                item = QTableWidgetItem(0)
                item.setData(0, f"{data.fnos[i]:d}")
                item.setData(7, 130)
                self.tw_meteor.setItem(i, 0, item)

                item = QTableWidgetItem(0)
                item.setData(0, f"{data.position_corrected.alt[i].value:.6f}°")
                item.setData(7, 130)
                self.tw_meteor.setItem(i, 1, item)

                item = QTableWidgetItem(0)
                item.setData(0, f"{data.position_corrected.az[i].value:.6f}°")
                item.setData(7, 130)
                self.tw_meteor.setItem(i, 2, item)

                item = QTableWidgetItem(0)
                item.setData(0, f"{data.magnitudes_corrected[i]:.6f}")
                item.setData(7, 130)
                self.tw_meteor.setItem(i, 3, item)
            # self.tw_meteor.setItem(i, 1, QTableWidgetItem(data.position_corrected.az[i].value))

    def maskSensor(self):
        errors = self.matcher.position_errors(self.projection, masked=False)
        self.matcher.mask_sensor_data(errors < np.radians(self.dsb_error_limit.value()))
        log.info(f"Culled the dots to {c.param(f'{self.dsb_error_limit.value():.3f}')}°: "
              f"{c.num(self.matcher.sensor_data.stars_pixels.count_valid)} are valid")
        self.onProjectionParametersChanged()
        self.showCounts()

    def maskCatalogueDistant(self):
        errors = self.matcher.position_errors_inverse(self.projection, masked=False)
        self.matcher.mask_catalogue(errors < np.radians(self.dsb_distance_limit.value()))
        log.info(f"Culled the catalogue to {self.dsb_distance_limit.value()}°: "
              f"{self.matcher.catalogue.count_valid} stars used")
        self.positionSkyPlot.invalidate_stars()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()
        self.showCounts()

    def maskCatalogueFaint(self):
        self.matcher.mask_catalogue(self.matcher.catalogue.vmag(masked=False) > self.dsb_magnitude_limit.value())
        log.info(f"Culled the catalogue to magnitude {self.dsb_magnitude_limit.value()}m: "
              f"{self.matcher.catalogue.count_valid} stars used")
        self.positionSkyPlot.invalidate_stars()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()
        self.showCounts()

    def resetValid(self):
        self.matcher.reset_mask()
        self.onProjectionParametersChanged()
        self.showCounts()

    @QtCore.pyqtSlot()
    def showCounts(self) -> None:
        if isinstance(self.matcher, Counselor):
            self.lb_mode.setText("paired")
            self.tab_correction_magnitudes_enabled.setEnabled(True)
        else:
            self.lb_mode.setText("unpaired")
            self.tab_correction_magnitudes_enabled.setEnabled(False)

        self.lb_catalogue_all.setText(f'{self.matcher.catalogue.count}')
        self.lb_catalogue_near.setText(f'{self.matcher.catalogue.count_valid}')
        self.lb_objects_all.setText(f'{self.matcher.sensor_data.stars.count}')
        self.lb_objects_near.setText(f'{self.matcher.sensor_data.stars.count_valid}')

    def exportCorrectedMeteor(self):
        if not self.paired:
            log.warn("Cannot export a meteor before pairing dots to the catalogue")
            return None

        filename, _ = QFileDialog.getSaveFileName(self, "Export corrected meteor to file", "output/",
                                                  "XML files (*.xml)")
        if filename is not None and filename != '':
            with open(filename, 'w') as file:
                file.write(f"""<?xml version="1.0" encoding="UTF-8" ?>
<ufoanalyzer_record version ="200"
    clip_name="{self.matcher.sensor_data.id}"
    o="1"
    y="{self.time.strftime("%Y")}"
    mo="{self.time.strftime("%m")}"
    d="{self.time.strftime("%d")}"
    h="{self.time.strftime("%H")}"
    m="{self.time.strftime("%M")}"
    s="{self.time.strftime('%S.%f')}"
    tz="0" tme="0" lid="{self.matcher.sensor_data.station}" sid="kvant"
    lng="{self.dsb_lon.value()}" lat="{self.dsb_lat.value()}" alt="{self.dsb_alt.value()}"
    cx="{self.matcher.sensor_data.rect.xmax}" cy="{self.matcher.sensor_data.rect.ymax}"
    fps="{self.matcher.sensor_data.fps}" interlaced="0" bbf="0"
    frames="{self.matcher.sensor_data.meteor.count}"
    head="{self.matcher.sensor_data.meteor.fnos(False)[0] - 1}"
    tail="0" drop="-1"
    dlev="0" dsize="0" sipos="0" sisize="0"
    trig="0" observer="{self.matcher.sensor_data.station}" cam="" lens=""
    cap="" u2="0" ua="0" memo=""
    az="0" ev="0" rot="0" vx="0"
    yx="0" dx="0" dy="0" k4="0"
    k3="0" k2="0" atc="0" BVF="0"
    maxLev="0" maxMag="0" minLev="0" mimMag="0"
    dl="0" leap="0" pixs="0" rstar="0"
    ddega="0" ddegm="0" errm="0" Lmrgn="0"
    Rmrgn="0" Dmrgn="0" Umrgn="0">
    <ua2_objects>
        <ua2_object
            fs="20" fe="64" fN="45" sN="45"
            sec="3" av="0" pix="0" bmax="0"
            bN="0" Lmax="0" mag="0" cdeg="0"
            cdegmax="0" io="0" raP="0" dcP="0"
            av1="0" x1="0" y1="0" x2="0"
            y2="0" az1="0" ev1="0" az2="0"
            ev2="0" azm="0" evm="0" ra1="0"
            dc1="0" ra2="0" dc2="0" ram="0"
            dcm="0" class="spo" m="0" dr="0"
            dv="0" Vo="0" lng1="0" lat1="0"
            h1="0" dist1="0" gd1="0" azL1="0"
            evL1="0" lng2="0" lat2="0" h2="0"
            dist2="0" gd2="0" len="0" GV="0"
            rao="0" dco="0" Voo="0" rat="0"
            dct="0" memo=""
            CodeRed="G"
            ACOM="324"
            sigma="0"
            sigma.azi="0"
            sigma.zen="0"
            A0="{self.projection.axis_shifter.a0}"
            X0="{self.projection.axis_shifter.x0}"
            Y0="{self.projection.axis_shifter.y0}"
            V="{self.projection.radial_transform.linear}"
            S="{self.projection.radial_transform.lin_coef}"
            D="{self.projection.radial_transform.lin_exp}"
            EPS="{self.projection.zenith_shifter.epsilon}"
            E="{self.projection.zenith_shifter.E}"
            A="{self.projection.axis_shifter.A}"
            F0="{self.projection.axis_shifter.F}"
            P="{self.projection.radial_transform.quad_coef}"
            Q="{self.projection.radial_transform.quad_exp}"
            C="1"
            CH1="0"
            CH2="0"
            CH3="0"
            CH4="0"
            magA="0"
            magB="0"
            magR2="0"
            magS="0"
            usingPrecession="False">
""")
                file.write(self.matcher.print_meteor(self.projection, self.calibration))
                file.write("""
        </ua2_object>
    </ua2_objects>
</ufoanalyzer_record>""")

    @property
    def grid_resolution(self):
        return self.sb_resolution.value()

    def pair(self):
        if (avg_error := np.degrees(self.matcher.avg_error(self.position_errors))) > 0.3:
            reply = QMessageBox.warning(self, "Mean position error limit exceeded!",
                                        f"Mean position error is currently {avg_error:.6f}°.\n"
                                        f"Are you sure your approximate solution is correct?",
                                        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if reply != QMessageBox.StandardButton.Ok:
                return False

        self.matcher = self.matcher.pair(self.projection)
        self.matcher.update_position_smoother(self.projection, bandwidth=self.bandwidth())
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=self.bandwidth())

        self.positionSkyPlot.invalidate_dots()
        self.positionSkyPlot.invalidate_stars()
        self.magnitudeSkyPlot.invalidate_dots()
        self.magnitudeSkyPlot.invalidate_stars()
        self.positionErrorPlot.invalidate()
        self.magnitudeErrorPlot.invalidate()
        self.positionCorrectionPlot.invalidate()
        self.magnitudeCorrectionPlot.invalidate()
        self.showCounts()
        self.updatePlots()

    def displayAbout(self):
        msg = QMessageBox(self, text="VASCO Virtual All-Sky CorrectOr plate")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("About")
        msg.setModal(True)
        msg.setInformativeText(f"Version {VERSION}, built on {DATE}")
        msg.move((self.width() - msg.width()) // 2, (self.height() - msg.height()) // 2)
        return msg.exec()
