import logging

import math
import yaml
import pytz
import datetime
import numpy as np
import dotmap

from PyQt6 import QtCore
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6.QtCore import QDateTime, Qt, QSignalBlocker

from astropy import units as u
from astropy.coordinates import EarthLocation
from pathlib import Path
import matplotlib as mpl
from astropy.time import Time

from amosutils.projections import BorovickaProjection

from models import Matcher
from models.qcataloguemodel import QCatalogueModel
from models.qstarmodel import QStarModel
from plotting import MainWindowPlots
from models import SensorData, QMeteorModel
from export import XMLExporter

import colour as c
from amos import AMOS, Station

mpl.use('Qt5Agg')

log = logging.getLogger('vasco')

VERSION = "0.9.1"
DATE = "2025-03-12"

np.set_printoptions(edgeitems=5, linewidth=256, formatter=dict(float=lambda x: f"{x:.6f}"))


class MainWindow(MainWindowPlots):
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.populate_stations()

        self.setup_parameters()
        self.update_location()
        self.update_time()
        self.reset_matcher()

        if args.catalogue:
            self.matcher.load_catalogue(args.catalogue.name)
        if args.sighting:
            self._load_sighting(args.sighting.name)
            self.update_time()
            self.update_location()
        if args.projection:
            self._import_projection_parameters(args.projection.name)
            self.update_scaling()
            self.update_projection()

        self.update_matcher()

        self.compute_position_errors()
        self.compute_magnitude_errors()

        self.update_plots()
        self.on_projection_parameters_changed()
        self.show_counts()

        self.connect_signal_slots()
        self.tw_charts.setCurrentIndex(2)

    def setup_parameters(self):
        self.pw_x0.setup(title="H shift", symbol="x<sub>0</sub>", unit="mm",
                         minimum=-10, maximum=10, step=0.001, initial_value=0)
        self.pw_y0.setup(title="V shift", symbol="y<sub>0</sub>", unit="mm",
                         minimum=-10, maximum=10, step=0.001, initial_value=0)
        self.pw_a0.setup(title="rotation", symbol="a<sub>0</sub>", unit="°",
                         minimum=0, maximum=359.999999, step=0.2, initial_value=0,
                         display_to_true=np.radians, true_to_display=np.degrees)

        self.pw_A.setup(title="amplitude", symbol="A", unit="",
                        minimum=-1, maximum=1, step=0.001)
        self.pw_F.setup(title="phase", symbol="F", unit="°", minimum=0, maximum=359.999999, step=1,
                        display_to_true=np.radians, true_to_display=np.degrees)

        self.pw_V.setup(title="linear", symbol="&V", unit="rad/mm",
                        minimum=0.001, maximum=1, step=0.001, initial_value=0.5)
        self.pw_S.setup(title="exp coef", symbol="&S", unit="rad/mm",
                        minimum=-100, maximum=100, step=0.001)
        self.pw_D.setup(title="exp exp", symbol="&D", unit="mm<sup>-1</sup>",
                        minimum=-100, maximum=100, step=0.0001)
        self.pw_P.setup(title="biexp coef", symbol="&P", unit="rad/mm",
                        minimum=-100, maximum=100, step=0.001)
        self.pw_Q.setup(title="biexp exp", symbol="&Q", unit="mm<sup>-2</sup>",
                        minimum=-100, maximum=100, step=0.0001)

        self.pw_epsilon.setup(title="zenith angle", symbol="ε", unit="°", minimum=0, maximum=90, step=0.1,
                              display_to_true=np.radians, true_to_display=np.degrees)
        self.pw_E.setup(title="azimuth", symbol="E", unit="°", minimum=0, maximum=359.999999, step=1,
                        display_to_true=np.radians, true_to_display=np.degrees)

        for widget in self.param_widgets.values():
            widget.dsb_value.valueChanged.connect(self.on_projection_parameters_changed)

    def connect_signal_slots(self):
        self.ac_load_sighting.triggered.connect(self.load_sighting)
        self.ac_load_catalogue.triggered.connect(self.load_catalogue)
        self.ac_export_meteor.triggered.connect(self.export_corrected_meteor)
        self.ac_mask_unmatched.triggered.connect(self.mask_sensor_dist)
        self.ac_create_pairing.triggered.connect(self.on_pair_clicked)
        self.ac_load_parameters.triggered.connect(self.import_projection_parameters)
        self.ac_save_parameters.triggered.connect(self.export_projection_parameters)
        self.ac_optimize.triggered.connect(self.optimize)
        self.ac_about.triggered.connect(self.display_about)

        self.cb_stations.currentIndexChanged.connect(self.select_station)
        self.dt_time.dateTimeChanged.connect(self.on_time_changed)

        self.dsb_lat.valueChanged.connect(self.on_location_changed)
        self.dsb_lon.valueChanged.connect(self.on_location_changed)
        self.dsb_xs.valueChanged.connect(self.on_scaling_changed)
        self.dsb_ys.valueChanged.connect(self.on_scaling_changed)

        # Parameter global interface
        self.pb_optimize.clicked.connect(self.optimize)
        self.sb_maxiter.valueChanged.connect(self.on_maxiter_changed)
        self.pb_pair.clicked.connect(self.on_pair_clicked)
        self.pb_import.clicked.connect(self.import_projection_parameters)
        self.pb_export.clicked.connect(self.export_projection_parameters)

        self.pb_import.setIcon(QIcon.fromTheme('document-open'))
        self.pb_export.setIcon(QIcon.fromTheme('document-save'))

        # Sensor operations
        self.dsb_sensor_limit_dist.valueChanged.connect(self.on_error_limit_changed)
        self.dsb_sensor_limit_alt.valueChanged.connect(self.on_error_limit_changed)
        self.pb_sensor_mask_dist.clicked.connect(self.mask_sensor_dist)
        self.pb_sensor_mask_alt.clicked.connect(self.mask_sensor_alt)
        self.pb_sensor_mask_reset.clicked.connect(self.reset_sensor_mask)
        self.pb_sensor_mask_reset.setIcon(QIcon.fromTheme('edit-clear'))

        # Catalogue operations
        self.dsb_catalogue_limit_dist.valueChanged.connect(self.on_error_limit_changed)
        self.dsb_catalogue_limit_mag.valueChanged.connect(self.on_error_limit_changed)
        self.dsb_catalogue_limit_alt.valueChanged.connect(self.on_error_limit_changed)
        self.pb_catalogue_mask_dist.clicked.connect(self.mask_catalogue_dist)
        self.pb_catalogue_mask_mag.clicked.connect(self.mask_catalogue_mag)
        self.pb_catalogue_mask_alt.clicked.connect(self.mask_catalogue_alt)
        self.pb_catalogue_mask_reset.clicked.connect(self.reset_catalogue_mask)
        self.pb_catalogue_mask_reset.setIcon(QIcon.fromTheme('edit-clear'))

        # Smoother
        self.hs_bandwidth.actionTriggered.connect(self.on_bandwidth_setting_changed)
        self.hs_bandwidth.sliderMoved.connect(self.on_bandwidth_setting_changed)
        self.hs_bandwidth.actionTriggered.connect(self.on_bandwidth_changed)
        self.hs_bandwidth.sliderReleased.connect(self.on_bandwidth_changed)
        self.sb_arrow_scale.valueChanged.connect(self.on_arrow_scale_changed)
        self.sb_resolution.valueChanged.connect(self.on_resolution_changed)

        self.cb_show_errors.clicked.connect(self.plot_position_correction_errors)
        self.cb_show_grid.clicked.connect(self.plot_position_correction_grid)
        self.cb_interpolation.currentIndexChanged.connect(self.plot_magnitude_correction_grid)

        self.pb_export_xml.clicked.connect(self.ac_export_meteor.trigger)

        self.tw_charts.currentChanged.connect(self.update_plots)
        log.debug(f"Signals and slots connected")

    def populate_stations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        log.debug(f"Populated stations: {len(self.cb_stations)}")

    def select_station(self, index):
        if index == 0:
            station = Station(0, "c", "custom", self.dsb_lat.value(), self.dsb_lon.value(), self.dsb_alt.value())
        else:
            station = list(AMOS.stations.values())[index - 1]

        self.dsb_lat.setValue(station.latitude)
        self.dsb_lon.setValue(station.longitude)
        self.dsb_alt.setValue(station.altitude)

        self.update_matcher()
        self.on_location_changed()

    def on_time_changed(self):
        self.update_time()
        self.on_location_time_changed()

    def on_location_changed(self):
        self.update_location()
        self.on_location_time_changed()

    def set_location(self, lat, lon, alt):
        self.dsb_lat.setValue(lat)
        self.dsb_lon.setValue(lon)
        self.dsb_alt.setValue(alt)

    def update_location(self):
        self.location = EarthLocation(
            self.dsb_lon.value() * u.deg,
            self.dsb_lat.value() * u.deg,
            self.dsb_alt.value() * u.m,
        )
        log.debug(f"Updated location to {self.location.geodetic}")

    def set_time(self, time):
        self.dt_time.setDateTime(QDateTime(time.date(), time.time(), Qt.TimeSpec.UTC))

    def update_time(self):
        self.time = Time(self.dt_time.dateTime().toPyDateTime())
        log.debug(f"Updated the time to {self.time}")

    def update_scaling(self):
        self.matcher.sensor_data.set_shifter_scales(
            self.dsb_xs.value() / 1000,
            self.dsb_ys.value() / 1000
        )

    def on_scaling_changed(self):
        self.update_scaling()
        self.on_projection_parameters_changed()

    def update_matcher(self):
        log.info(f"Time / location changed: {self.time}, {self.location}")
        self.matcher.update_location_time(self.location, Time(self.time))
        self.matcher.update_position_smoother()

    def update_projection(self):
        log.info(f"Projection parameters changed: {self.get_projection_parameters()}")
        self.projection = BorovickaProjection(*self.get_projection_parameters())
        self.matcher.update_projection(self.projection)

    def on_projection_parameters_changed(self):
        self.update_projection()

        self.position_sky_plot.invalidate_dots()
        self.position_sky_plot.invalidate_meteor()
        self.magnitude_sky_plot.invalidate_dots()
        self.magnitude_sky_plot.invalidate_meteor()
        self.position_error_plot.invalidate()
        self.magnitude_error_plot.invalidate()
        self.position_correction_plot.invalidate()
        self.magnitude_correction_plot.invalidate()
        self.matcher.update_position_smoother(bandwidth=self.bandwidth())

        self.compute_position_errors()
        self.compute_magnitude_errors()
        self.update_plots()

    def on_location_time_changed(self):
        self.update_matcher()
        self.position_sky_plot.invalidate_stars()
        self.position_sky_plot.invalidate_dots()
        self.magnitude_sky_plot.invalidate_stars()
        self.magnitude_sky_plot.invalidate_dots()
        self.position_error_plot.invalidate()
        self.magnitude_error_plot.invalidate()

        bandwidth = self.bandwidth()
        self.matcher.update_position_smoother(bandwidth=bandwidth)
        self.matcher.update_magnitude_smoother(self.calibration, bandwidth=bandwidth)
        self.position_correction_plot.invalidate()
        self.magnitude_correction_plot.invalidate()

        self.compute_position_errors()
        self.compute_magnitude_errors()
        self.update_plots()

    def on_error_limit_changed(self):
        self.position_sky_plot.invalidate_dots()
        self.position_error_plot.invalidate()
        self.position_correction_plot.invalidate_dots()
        self.update_plots()

    def on_bandwidth_setting_changed(self):
        bandwidth = self.bandwidth()
        self.lb_bandwidth.setText(f"{bandwidth:.03f}")

    def on_bandwidth_changed(self, action=0):
        if action == 7:  # do not do anything if the user did not drop the slider yet
            return

        bandwidth = self.bandwidth()
        self.matcher.update_position_smoother(bandwidth=bandwidth)
        self.matcher.update_magnitude_smoother(self.calibration, bandwidth=bandwidth)
        self.position_correction_plot.invalidate_grid()
        self.position_correction_plot.invalidate_meteor()
        self.magnitude_correction_plot.invalidate_grid()
        self.magnitude_correction_plot.invalidate_meteor()
        self.update_plots()

    def on_arrow_scale_changed(self):
        self.position_correction_plot.invalidate_dots()
        self.position_correction_plot.invalidate_meteor()
        self.update_plots()

    def on_resolution_changed(self):
        self.position_correction_plot.invalidate_grid()
        self.magnitude_correction_plot.invalidate_grid()
        self.update_plots()

    def on_maxiter_changed(self):
        self.pg_optimize.setValue(0)
        self.pg_optimize.setMaximum(self.sb_maxiter.value())

    def bandwidth(self):
        return 10**(-self.hs_bandwidth.value() / 100)

    def reset_matcher(self):
        self.matcher = Matcher(self.location, self.time)

    def compute_position_errors(self):
        log.debug("Recomputing position errors...")
        self.position_errors = self.matcher.position_errors_sky(1, mask_catalogue=True, mask_sensor=True)

    def compute_magnitude_errors(self):
        log.debug("Recomputing magnitude errors...")
        self.magnitude_errors = self.matcher.magnitude_errors_sky(self.calibration)

    def load_catalogue(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load catalogue file", "catalogues",
                                                  "Tab-separated values (*.tsv)")
        filename = Path(filename)
        if filename == '':
            log.warning("No file provided, loading aborted")
        else:
            try:
                self.matcher.load_catalogue(filename)
            except:
                log.error(f"Could not load catalogue: {filename}")
            self.position_sky_plot.invalidate_stars()
            self.magnitude_sky_plot.invalidate_stars()
            self.on_projection_parameters_changed()

    def export_projection_parameters(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export projection parameters to file", "calibrations",
                                                  "YAML files (*.yaml)")
        if filename is not None and filename != '':
            self._export_projection_parameters(filename)

    def _export_projection_parameters(self, filename):
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

    def load_sighting(self):
        self._block_parameter_signals(True)
        self._block_location_time_signals(True)
        self._block_pixel_scales_signals(True)

        filename, _ = QFileDialog.getOpenFileName(self, "Load Kvant YAML file", "data",
                                                  "YAML files (*.yml *.yaml)")

        if filename == '':
            log.warning("No file provided, loading aborted")
        else:
            self._load_sighting(filename)

        if (station := AMOS.stations.get(self.matcher.sensor_data.station, None)) is not None:
            log.info(f"Position for station {station.code} found, loading properties from AMOS database")
            self.cb_stations.setCurrentIndex(station.id)
            if (path := Path(f'./calibrations/{station.code}.yaml')).exists():
                self._import_projection_parameters(path)
                log.info(f"Calibration file {path} found, loading projection parameters")
            else:
                log.info(f"Calibration file {path} not found, skipping")
        else:
            log.warning(f"Station not found in the database, marking as custom coordinates")
            self.cb_stations.setCurrentIndex(0)

        self._block_parameter_signals(False)
        self._block_location_time_signals(False)
        self._block_pixel_scales_signals(False)
        self.on_location_time_changed()
        self.on_projection_parameters_changed()
        self.sensor_plot.invalidate()
        self.update_plots()

    def _load_sighting(self, file):
        data = dotmap.DotMap(yaml.safe_load(open(file, 'r')), _dynamic=False)
        self.set_location(data.Latitude, data.Longitude, data.Altitude)
        self.set_time(pytz.UTC.localize(datetime.datetime.strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f")))
        self.sensor_plot.invalidate()

        self.matcher.sensor_data = SensorData.load_YAML(file)

        log.info(f"Loaded a sighting from {file}: "
                 f"{self.matcher.sensor_data.stars.count} stars, "
                 f"{self.matcher.sensor_data.meteor.count} frames")

    def import_projection_parameters(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import projection parameters from file", "../calibrations",
                                                  "YAML files (*.yml *.yaml)")
        self._import_projection_parameters(Path(filename))
        self.on_projection_parameters_changed()

    def _import_projection_parameters(self, filename: Path):
        try:
            self._block_parameter_signals(True)
            with open(filename, 'r') as file:
                try:
                    data = dotmap.DotMap(yaml.safe_load(file), _dynamic=False)
                    for param, widget in self.param_widgets.items():
                        widget.set_display_value(widget.true_to_display(data.projection.parameters[param]))
                        self.dsb_xs.setValue(data.pixels.xs)
                        self.dsb_ys.setValue(data.pixels.ys)
                    log.info(f"Imported projection parameters from {filename}")
                except yaml.YAMLError as exc:
                    log.error(f"Could not parse file {filename} as YAML: {exc}")
        except FileNotFoundError:
            log.error(f"File not found: {filename}")
        except Exception as exc:
            log.error(f"Could not import projection parameters: {exc}")
        finally:
            self._block_parameter_signals(False)

    def _block_parameter_signals(self, block: bool) -> None:
        for widget in self.param_widgets.values():
            widget.dsb_value.blockSignals(block)

    def _block_location_time_signals(self, block: bool) -> None:
        for widget in self.location_time_widgets.values():
            widget.blockSignals(block)

    def _block_pixel_scales_signals(self, block: bool) -> None:
        for widget in self.pixel_scale_widgets.values():
            widget.blockSignals(block)

    def get_projection_parameters(self):
        return np.array([widget.true_value for widget in self.param_widgets.values()], dtype=float)

    def optimize(self) -> None:
        self.w_input.setEnabled(False)
        self.w_input.repaint()

        iters = 0
        maxiters = self.sb_maxiter.value()
        original_title = self.gb_identify.title()

        def callback(x):
            nonlocal iters
            log.debug(x)
            iters += 1
            self.pg_optimize.setValue(iters)

        result = self.matcher.minimize(
            x0=self.get_projection_parameters(),
            maxiter=maxiters,
            mask=np.array([widget.is_checked() for widget in self.param_widgets.values()], dtype=bool),
            callback=callback,
        )

        self._block_parameter_signals(True)
        for value, widget in zip(result, self.param_widgets.values()):
            widget.set_true_value(value)
        self._block_parameter_signals(False)

        self.pg_optimize.setValue(maxiters)
        self.gb_identify.setTitle(original_title)
        self.w_input.setEnabled(True)
        self.w_input.repaint()
        self.on_projection_parameters_changed()

    def update_stars_table(self):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=False)
        x = self.matcher.sensor_data.stars.xs(masked=False)
        y = self.matcher.sensor_data.stars.ys(masked=False)
        shifted = self.matcher.sensor_data.shifter.invert(x, y)

        data = dotmap.DotMap(
            x=x,
            y=y,
            px=shifted[0],
            py=shifted[1],
            alt=np.degrees(positions[..., 0]),
            az=np.degrees(positions[..., 1]),
            star=self.matcher.pairing,
            mask=self.matcher.sensor_data.stars.mask,
            count=self.matcher.sensor_data.stars.count,
            scalar_errors=np.degrees(self.matcher.distance_sky(masked=False)),
            vector_errors=np.degrees(self.matcher.vector_errors_full()),
            _dynamic=False,
        )

        model = QStarModel(data)
        self.tv_sensor.setModel(model)

        for i, width in enumerate([120, 120, 120, 120, 120, 120, 80]):
            self.tv_sensor.setColumnWidth(i, width)

    def update_meteor_table(self):
        data = self.matcher.correct_meteor(self.projection, self.calibration)
        model = QMeteorModel(data)
        self.tv_meteor.setModel(model)
        for i, width in enumerate([40, 120, 120, 120, 120, 120, 120, 120, 120, 120]):
            self.tv_meteor.setColumnWidth(i, width)

    def update_catalogue_table(self):
        radec = self.matcher.catalogue.radec(self.location, self.time, masked=False)
        altaz = self.matcher.catalogue.altaz(self.location, self.time, masked=False)
        vmag = self.matcher.catalogue.vmag(self.location, self.time, masked=False)
        data = dotmap.DotMap(
            dec=radec.dec.degree,
            ra=radec.ra.degree,
            alt=altaz.alt.degree,
            az=altaz.az.degree,
            vmag=vmag,
            mask=self.matcher.catalogue.mask,
            count=self.matcher.catalogue.count,
            _dynamic=False,
        )

        model = QCatalogueModel(data)
        self.tv_catalogue.setModel(model)

        for i, width in enumerate([40, 120, 120, 120, 120, 80]):
            self.tv_catalogue.setColumnWidth(i, width)

    def reset_sensor_mask(self):
        log.debug("Reset the sensor mask")
        self.matcher.sensor_data.reset_mask()
        self.on_projection_parameters_changed()
        self.show_counts()

    def mask_sensor_dist(self):
        errors = self.matcher.distance_sky_full()
        errors = np.min(errors, axis=1, initial=math.tau / 2)
        limit = self.dsb_sensor_limit_dist.value()
        self._mask_sensor(errors < np.radians(limit), f"position errors < {c.num(f'{limit:.3f}°')}")

    def mask_sensor_alt(self):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=False, flip_theta=True)
        limit = self.dsb_sensor_limit_alt.value()
        self._mask_sensor(positions[..., 0] > np.radians(limit), f"altitude < {c.num(f'{limit:.1f}°')}")

    def _mask_sensor(self, mask: np.ndarray, message: str):
        self.matcher.mask_sensor_data(mask)
        log.info(f"Masked reference dots: {message}: "
                 f"{c.num(self.matcher.sensor_data.stars.count_visible)} are valid")
        self.matcher.update_pairing()
        self._update_catalogue_mask()

    def _update_catalogue_mask(self):
        """
        Update everything after the catalogue mask was changed
        """
        self.position_sky_plot.invalidate_stars()
        self.position_sky_plot.invalidate_dots()
        self.magnitude_sky_plot.invalidate_stars()
        self.magnitude_sky_plot.invalidate_dots()
        self.position_error_plot.invalidate()
        self.magnitude_error_plot.invalidate()

        self.compute_position_errors()
        self.compute_magnitude_errors()
        self.update_plots()
        self.show_counts()

    def _mask_catalogue(self, mask: np.ndarray, message: str):
        self.matcher.mask_catalogue(mask)
        log.info(f"Masked the catalogue: {message}: "
                 f"{c.num(self.matcher.catalogue.count_visible)} stars visible")
        self._update_catalogue_mask()
        self.matcher.update_pairing()

    def mask_catalogue_dist(self):
        errors: np.ndarray = np.min(self.matcher.distance_sky_full(), axis=0)
        limit: float = self.dsb_catalogue_limit_dist.value()
        self._mask_catalogue(errors < np.radians(limit),
                             f"errors < {c.num(f'{limit:.3f}°')}")

    def mask_catalogue_mag(self):
        limit: float = self.dsb_catalogue_limit_mag.value()
        self._mask_catalogue(self.matcher.catalogue_vmag(masked=False) < limit,
                             f"magnitude <{c.num(f'{limit:.1f}m')}")

    def mask_catalogue_alt(self):
        limit: float = self.dsb_catalogue_limit_alt.value()
        self._mask_catalogue(self.matcher.catalogue_altaz_np(masked=False)[..., 0] > np.radians(limit),
                             f"altitude >{c.num(f'{limit:.1f}°')}")

    def reset_catalogue_mask(self):
        log.debug("Reset the catalogue mask")
        self.matcher.catalogue.mask = None
        self.on_projection_parameters_changed()
        self.on_location_changed()
        self.show_counts()

    @QtCore.pyqtSlot()
    def show_counts(self) -> None:
        # FixMe make this depend on pairing
        paired = False
        self.lb_mode.setText(f"{'' if paired else 'un'}paired")

        self.lb_catalogue_total.setText(f'{self.matcher.catalogue.count}')
        self.lb_catalogue_used.setText(f'{self.matcher.catalogue.count_visible}')
        self.lb_sensor_total.setText(f'{self.matcher.sensor_data.stars.count}')
        self.lb_sensor_used.setText(f'{self.matcher.sensor_data.stars.count_visible}')

    def export_corrected_meteor(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export corrected meteor to file", "../output/",
                                                  "XML files (*.xml)")
        if filename is not None and filename != '':
            exporter = XMLExporter(self.matcher, self.location, self.time, self.projection, self.calibration)
            exporter.export(filename)

    @property
    def grid_resolution(self):
        return self.sb_resolution.value()

    def on_pair_clicked(self):
        self.matcher.fix_pairing(not self.matcher.pairing_fixed)
        self.lb_mode.setText(f"{'' if self.matcher.pairing_fixed else 'un'}paired")

    def display_about(self):
        msg = QMessageBox(self, text="VASCO Virtual All-Sky CorrectOr plate")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("About")
        msg.setModal(True)
        msg.setInformativeText(f"Version {VERSION}, {DATE}")
        msg.move((self.width() - msg.width()) // 2, (self.height() - msg.height()) // 2)
        return msg.exec()
