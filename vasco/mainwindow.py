import logging

import yaml
import pytz
import datetime
import numpy as np
import dotmap

from PyQt6 import QtCore
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QDateTime, Qt

from astropy import units as u
from astropy.coordinates import EarthLocation
from pathlib import Path
import matplotlib as mpl
from astropy.time import Time

from matchers import Matchmaker, Counsellor
from amosutils.projections import BorovickaProjection
from plotting import MainWindowPlots
from models import SensorData, QMeteorModel
from export import XMLExporter

import colour as c
from amos import AMOS, Station

mpl.use('Qt5Agg')

log = logging.getLogger('vasco')

VERSION = "0.8.0"
DATE = "2023-07-18"


class MainWindow(MainWindowPlots):
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.populateStations()
        self.updateProjection()

        self.connectSignalSlots()

        self.updateLocation()
        self.updateTime()

        self.resetMatcher()
        if args.catalogue:
            self.matcher.load_catalogue(args.catalogue.name)
        if args.sighting:
            self._loadSighting(args.sighting.name)
        if args.projection:
            self._importProjectionParameters(args.projection.name)

        self.showCounts()
        self.onProjectionParametersChanged()
        self.onScalingChanged()

        self.tw_charts.setCurrentIndex(1)

    def connectSignalSlots(self):
        self.ac_load_sighting.triggered.connect(self.loadSighting)
        self.ac_load_catalogue.triggered.connect(self.loadCatalogue)
        self.ac_export_meteor.triggered.connect(self.exportCorrectedMeteor)
        self.ac_mask_unmatched.triggered.connect(self.maskSensor)
        self.ac_create_pairing.triggered.connect(self.pair)
        self.ac_load_parameters.triggered.connect(self.importProjectionParameters)
        self.ac_save_parameters.triggered.connect(self.exportProjectionParameters)
        self.ac_optimize.triggered.connect(self.optimize)
        self.ac_about.triggered.connect(self.displayAbout)

        for widget in self.param_widgets.values():
            widget.dsb_value.valueChanged.connect(self.onProjectionParametersChanged)

        "The shape of the dot collection and the catalogue must be the same, got {obs.shape} and {cat.shape}"
        self.pw_x0.setup(title="H shift", symbol="x<sub>0</sub>", unit="µm", minimum=-10, maximum=10, step=0.001)
        self.pw_y0.setup(title="V shift", symbol="y<sub>0</sub>", unit="µm", minimum=-10, maximum=10, step=0.001)
        self.pw_a0.setup(title="rotation", symbol="a<sub>0</sub>", unit="°", minimum=0, maximum=359.999999, step=0.2,
                         display_to_true=np.radians, true_to_display=np.degrees)

        self.pw_A.setup(title="amplitude", symbol="A", unit="", minimum=-1, maximum=1, step=0.001)
        self.pw_F.setup(title="phase", symbol="F", unit="°", minimum=0, maximum=359.999999, step=1,
                        display_to_true=np.radians, true_to_display=np.degrees)

        self.pw_V.setup(title="linear", symbol="&V", unit="rad/mm", minimum=0.001, maximum=1, step=0.001)
        self.pw_S.setup(title="exp coef", symbol="&S", unit="rad/mm", minimum=-100, maximum=100, step=0.001)
        self.pw_D.setup(title="exp exp", symbol="&D", unit="mm<sup>-1</sup>", minimum=-100, maximum=100, step=0.0001)
        self.pw_P.setup(title="biexp coef", symbol="&P", unit="rad/mm", minimum=-100, maximum=100, step=0.001)
        self.pw_Q.setup(title="biexp exp", symbol="&Q", unit="mm<sup>-2</sup>", minimum=-100, maximum=100, step=0.0001)

        self.pw_epsilon.setup(title="zenith angle", symbol="ε", unit="°", minimum=0, maximum=90, step=0.1,
                              display_to_true=np.radians, true_to_display=np.degrees)
        self.pw_E.setup(title="azimuth", symbol="E", unit="°", minimum=0, maximum=359.999999, step=1,
                        display_to_true=np.radians, true_to_display=np.degrees)

        self.dt_time.dateTimeChanged.connect(self.updateTime)
        self.dt_time.dateTimeChanged.connect(self.onTimeChanged)

        self.dsb_lat.valueChanged.connect(self.onLocationChanged)
        self.dsb_lon.valueChanged.connect(self.onLocationChanged)
        self.dsb_xs.valueChanged.connect(self.onScalingChanged)
        self.dsb_ys.valueChanged.connect(self.onScalingChanged)

        self.pb_optimize.clicked.connect(self.optimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.exportProjectionParameters)
        self.pb_import.clicked.connect(self.importProjectionParameters)

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

        self.pb_export_xml.clicked.connect(self.ac_export_meteor.trigger)

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
        self.matcher.update(self.location, Time(self.time))
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
        if action == 7:  # do not do anything if the user did not drop the slider yet
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
        filename, _ = QFileDialog.getOpenFileName(self, "Load catalogue file", "../catalogues",
                                                  "Tab-separated values (*.tsv)")
        if filename == '':
            log.warning("No file provided, loading aborted")
        else:
            if self.paired:
                self.resetMatcher()
            self.matcher.load_catalogue(filename)
            self.positionSkyPlot.invalidate_stars()
            self.magnitudeSkyPlot.invalidate_stars()
            self.onProjectionParametersChanged()

    def exportProjectionParameters(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export projection parameters to file", "../calibrations",
                                                  "YAML files (*.yaml)")
        if filename is not None and filename != '':
            self._exportProjectionParameters(filename)

    def _exportProjectionParameters(self, filename):
        try:
            with open(filename, 'w+') as file:
                yaml.dump(dict(
                    projection=dict(
                        name='Borovička',
                        parameters={param: widget.true_value for param, widget in self.param_widgets.items()},
                    ),
                    pixels=dict(xs=self.dsb_xs.value(), ys=self.dsb_ys.value()),
                ), file)
        except FileNotFoundError as exc:
            log.error(f"Could not export projection parameters: {exc}")

    def loadSighting(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Kvant YAML file", "../data",
                                                  "YAML files (*.yml *.yaml)")
        if filename == '':
            log.warning("No file provided, loading aborted")
        else:
            self._loadSighting(filename)

        if (station := AMOS.stations.get(self.matcher.sensor_data.station, None)) is not None:
            log.info(f"Position for station {station.code} found, loading properties from AMOS database")
            self.cb_stations.setCurrentIndex(station.id)
            if (path := Path(f'./calibrations/{station.code}.yaml')).exists():
                self._importProjectionParameters(path)
                log.info(f"Calibration file {path} found, loading projection parameters")
            else:
                log.info(f"Calibration file {path} not found, skipping")
        else:
            log.warning(f"Station not found in the database, marking as custom coordinates")
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
        self.matcher.sensor_data = SensorData.load_YAML(file)

    def importProjectionParameters(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import projection parameters from file", "../calibrations",
                                                  "YAML files (*.yml *.yaml)")
        self._importProjectionParameters(filename)
        self.onProjectionParametersChanged()

    def _importProjectionParameters(self, filename):
        try:
            self._blockParameterSignals(True)
            with open(filename, 'r') as file:
                try:
                    data = dotmap.DotMap(yaml.safe_load(file), _dynamic=False)
                    for param, widget in self.param_widgets.items():
                        widget.set_display_value(widget.true_to_display(data.projection.parameters[param]))
                        self.dsb_xs.setValue(data.pixels.xs)
                        self.dsb_ys.setValue(data.pixels.ys)
                except yaml.YAMLError as exc:
                    log.error(f"Could not parse file {filename} as YAML: {exc}")
        except FileNotFoundError:
            log.error(f"File not found: {filename}")
        except Exception as exc:
            log.error(f"Could not import projection parameters: {exc}")
        finally:
            self._blockParameterSignals(False)
            self.updateProjection()

    def _blockParameterSignals(self, block: bool) -> None:
        for widget in self.param_widgets.values():
            widget.dsb_value.blockSignals(block)

    def getProjectionParameters(self):
        return np.array([widget.true_value for widget in self.param_widgets.values()], dtype=float)

    def optimize(self) -> None:
        self.w_input.setEnabled(False)
        self.w_input.repaint()

        result = self.matcher.minimize(
            x0=self.getProjectionParameters(),
            maxiter=self.sb_maxiter.value(),
            mask=np.array([widget.is_checked() for widget in self.param_widgets.values()], dtype=bool)
        )

        self._blockParameterSignals(True)
        for value, widget in zip(result, self.param_widgets.values()):
            widget.set_true_value(value)
        self._blockParameterSignals(False)

        self.w_input.setEnabled(True)
        self.w_input.repaint()
        self.onProjectionParametersChanged()

    def updateMeteorTable(self):
        if self.paired:
            self.tabs_table.setCurrentIndex(1)
            data = self.matcher.correct_meteor(self.projection, self.calibration)
            model = QMeteorModel(data)
            self.tv_meteor.setModel(model)
            for i, width in enumerate([40, 160, 160, 160, 160, 160, 160, 160, 160, 160]):
                self.tv_meteor.setColumnWidth(i, width)
        else:
            self.tabs_table.setCurrentIndex(0)

    def maskSensor(self):
        if self.paired:
            self.pair()

        errors = self.matcher.position_errors(self.projection, masked=False)
        self.matcher.mask_sensor_data(errors < np.radians(self.dsb_error_limit.value()))
        log.info(f"Culled the dots to {c.param(f'{self.dsb_error_limit.value():.3f}')}°: "
                 f"{c.num(self.matcher.sensor_data.stars.count_valid)} are valid")
        self.onProjectionParametersChanged()
        self.showCounts()

    def maskCatalogueDistant(self):
        if self.paired:
            self.pair()

        errors = self.matcher.position_errors_inverse(self.projection, masked=False)
        self.matcher.mask_catalogue(errors < np.radians(self.dsb_distance_limit.value()))
        log.info(f"Culled the catalogue to {c.num(f'{self.dsb_distance_limit.value():.3f}')}°: "
                 f"{c.num(self.matcher.catalogue.count_valid)} stars used")

        self.positionSkyPlot.invalidate_stars()
        self.magnitudeSkyPlot.invalidate_stars()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()
        self.showCounts()

    def maskCatalogueFaint(self):
        self.matcher.mask_catalogue(self.matcher.catalogue.vmag(masked=False) > self.dsb_magnitude_limit.value())
        log.info(f"Culled the catalogue to magnitude {self.dsb_magnitude_limit.value()}m: "
                 f"{self.matcher.catalogue.count_valid} stars used")

        if self.paired:
            self.pair()

        self.positionSkyPlot.invalidate_stars()
        self.magnitudeSkyPlot.invalidate_stars()

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
        if isinstance(self.matcher, Counsellor):
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
            log.warning("Cannot export a meteor before pairing dots to the catalogue")
            return None

        filename, _ = QFileDialog.getSaveFileName(self, "Export corrected meteor to file", "../output/",
                                                  "XML files (*.xml)")
        if filename is not None and filename != '':
            exporter = XMLExporter(self.matcher, self.location, self.time, self.projection, self.calibration)
            exporter.export(filename)

    @property
    def grid_resolution(self):
        return self.sb_resolution.value()

    def pair(self):
        if (rms_error := np.degrees(self.matcher.rms_error(self.position_errors))) > 0.3:
            reply = QMessageBox.warning(self, "Mean position error limit exceeded!",
                                        f"Mean position error is currently {rms_error:.6f}°.\n"
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
