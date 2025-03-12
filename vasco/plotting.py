import logging

import numpy as np

from typing import Callable

from PyQt6.QtWidgets import QStackedWidget
from astropy.time import Time

from plots import SensorPlot
from plots.sky import PositionSkyPlot, MagnitudeSkyPlot
from plots.errors import PositionErrorPlot, MagnitudeErrorPlot
from plots.correction import BaseCorrectionPlot, PositionCorrectionPlot, MagnitudeCorrectionPlot
from utilities import unit_grid

from base import MainWindowBase


log = logging.getLogger('vasco')


class MainWindowPlots(MainWindowBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensorPlot = SensorPlot(self.tab_sensor_plot)
        self.positionSkyPlot = PositionSkyPlot(self.tab_sky_positions)
        self.magnitudeSkyPlot = MagnitudeSkyPlot(self.tab_sky_magnitudes)
        self.positionErrorPlot = PositionErrorPlot(self.tab_errors_positions)
        self.magnitudeErrorPlot = MagnitudeErrorPlot(self.tab_errors_magnitudes)
        self.positionCorrectionPlot = PositionCorrectionPlot(self.tab_correction_positions_enabled)
        self.magnitudeCorrectionPlot = MagnitudeCorrectionPlot(self.tab_correction_magnitudes_enabled)

    def update_plots(self):
        """
        Iterate over all plots and update all that are no longer valid
        """
        self.show_errors()
        links = [
            [
                (self.sensorPlot.valid, self.plot_sensor_data),
            ],
            [],
            [
                (self.positionSkyPlot.valid_dots, self.plot_observed_stars_positions),
                (self.positionSkyPlot.valid_stars, self.plot_catalogue_stars_positions),
            ],
            [
                (self.magnitudeSkyPlot.valid_dots, self.plot_observed_stars_magnitudes),
                (self.magnitudeSkyPlot.valid_stars, self.plot_catalogue_stars_magnitudes),
            ],
            [
                (self.positionErrorPlot.valid_dots, self.plot_position_errors_dots),
                (self.positionErrorPlot.valid_meteor, self.plot_position_errors_meteor),
            ],
            [
                (self.magnitudeErrorPlot.valid_dots, self.plot_magnitude_errors_dots),
                (self.magnitudeErrorPlot.valid_meteor, self.plot_magnitude_errors_meteor),
            ],
            [
                (self.positionCorrectionPlot.valid_dots, self.plot_position_correction_errors),
                (self.positionCorrectionPlot.valid_meteor, self.plotPositionCorrectionMeteor),
                (self.positionCorrectionPlot.valid_grid, self.plot_position_correction_grid),
            ],
            [
                (self.magnitudeCorrectionPlot.valid_dots, self.plotMagnitudeCorrectionErrors),
                (self.magnitudeCorrectionPlot.valid_meteor, self.plotMagnitudeCorrectionMeteor),
                (self.magnitudeCorrectionPlot.valid_grid, self.plot_magnitude_correction_grid),
            ],
            [],
        ][self.tw_charts.currentIndex()]

        for valid, function in links:
            if not valid:
                function()

        self.update_sensor_table()
        self.update_meteor_table()

    """ Methods for plotting sensor data """

    def plot_sensor_data(self):
        self.sensorPlot.update(self.matcher.sensor_data)

    """ Methods for plotting sky charts """

    def _plot_observed_stars(self, plot, errors, *, limit=None):
        plot.update_dots(
            self.matcher.sensor_data.stars.project(self.projection, masked=True),
            self.matcher.sensor_data.stars.i,
            errors,
            limit=limit,
        )
        plot.update_meteor(
            self.matcher.sensor_data.meteor.project(self.projection, masked=True),
            self.matcher.sensor_data.meteor.i
        )

    def plot_observed_stars_positions(self):
        log.debug(f"Plotting dot positions")
        self._plot_observed_stars(self.positionSkyPlot, self.position_errors,
                                  limit=np.radians(self.dsb_sensor_limit_dist.value()))

    def plot_observed_stars_magnitudes(self):
        log.debug(f"Plotting dot magnitudes")
        self._plot_observed_stars(self.magnitudeSkyPlot, self.magnitude_errors,
                                  limit=np.radians(self.dsb_sensor_limit_dist.value()))

    def _plot_catalogue_stars(self, plot):
        plot.update_stars(
            self.matcher.altaz(masked=True),
            self.matcher.catalogue.vmag(self.location, Time(self.time), masked=True)
        )

    def plot_catalogue_stars_positions(self):
        log.debug(f"Plotting star positions")
        self._plot_catalogue_stars(self.positionSkyPlot)

    def plot_catalogue_stars_magnitudes(self):
        log.debug(f"Plotting star magnitudes")
        self._plot_catalogue_stars(self.magnitudeSkyPlot)

    """ Methods for plotting error charts """

    def _plot_errors_dots(self, plot, errors):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.stars.intensities(True)
        plot.update_dots(positions, magnitudes, errors, limit=self.dsb_sensor_limit_dist.value())

    def plot_position_errors_dots(self):
        log.debug(f"Plotting position errors")
        self._plot_errors_dots(self.positionErrorPlot, self.position_errors)

    def plot_magnitude_errors_dots(self):
        log.debug(f"Plotting magnitude errors")
        self._plot_errors_dots(self.magnitudeErrorPlot, self.magnitude_errors)

    def _plot_errors_meteor(self, plot, errors):
        positions = self.matcher.sensor_data.meteor.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.meteor.intensities(True)
        plot.update_meteor(positions, magnitudes, errors, limit=0.1)

    def plot_position_errors_meteor(self):
        if self.paired:
            xy = self.matcher.correction_meteor_xy(self.projection)
            correction = np.degrees(np.sqrt(xy[..., 0]**2 + xy[..., 1]**2))
            self._plot_errors_meteor(self.positionErrorPlot, correction)

    def plot_magnitude_errors_meteor(self):
        if self.paired:
            self._plot_errors_meteor(self.magnitudeErrorPlot, self.matcher.correction_meteor_mag(self.projection))

    """ Methods for updating correction plots """

    def _switch_tabs(self,
                     tabs: QStackedWidget,
                     func: Callable[[BaseCorrectionPlot, ...], None],
                     *args, **kwargs) -> None:
        if self.paired:
            tabs.setCurrentIndex(1)
            func(*args, **kwargs)
        else:
            tabs.setCurrentIndex(0)

    def _plotCorrectionErrors(self, plot: BaseCorrectionPlot) -> None:
        if self.cb_show_errors.isChecked():
            log.debug(f"Plotting {plot.intent} for {plot.target}")
            plot.update_dots(
                self.matcher.catalogue.altaz(self.location, self.time, masked=True),
                self.matcher.sensor_data.stars.project(self.projection, masked=True),
                self.matcher.catalogue.vmag(masked=True),
                self.matcher.sensor_data.stars.calibrate(self.calibration, masked=True),
                limit=np.radians(self.dsb_sensor_limit_dist.value()),
                scale=1 / self.sb_arrow_scale.value(),
            )
        else:
            plot.clear_errors()

    def plot_position_correction_errors(self) -> None:
        self._switch_tabs(self.tabs_positions, self._plotCorrectionErrors, self.positionCorrectionPlot)

    def plotMagnitudeCorrectionErrors(self) -> None:
        self._switch_tabs(self.tabs_magnitudes, self._plotCorrectionErrors, self.magnitudeCorrectionPlot)

    def _plotCorrectionMeteor(self, plot) -> None:
        plot.update_meteor(
            self.matcher.sensor_data.meteor.project(self.projection, masked=True),
            self.matcher.correction_meteor_xy(self.projection),
            self.matcher.sensor_data.meteor.calibrate(self.calibration, masked=True),
            self.matcher.correction_meteor_mag(self.projection),
            scale=1 / self.sb_arrow_scale.value(),
        )

    def plotPositionCorrectionMeteor(self) -> None:
        self._switch_tabs(self.tabs_positions, self._plotCorrectionMeteor, self.positionCorrectionPlot)

    def plotMagnitudeCorrectionMeteor(self) -> None:
        self._switch_tabs(self.tabs_magnitudes, self._plotCorrectionMeteor, self.magnitudeCorrectionPlot)

    def _plotCorrectionGrid(self, plot, grid, *, masked: bool, **kwargs):
        if self.cb_show_grid.isChecked():
            xx, yy = unit_grid(self.grid_resolution, masked=masked)
            plot.update_grid(xx, yy, grid(resolution=self.grid_resolution), **kwargs)
        else:
            plot.clear_grid()

    def plot_position_correction_grid(self):
        self._switch_tabs(
            self.tabs_positions,
            lambda plot: self._plotCorrectionGrid(plot, self.matcher.position_grid, masked=True),
            self.positionCorrectionPlot,
        )

    def plot_magnitude_correction_grid(self):
        self._switch_tabs(
            self.tabs_magnitudes,
            lambda plot: self._plotCorrectionGrid(plot, self.matcher.magnitude_grid, masked=False,
                                                  interpolation=self.cb_interpolation.currentText()),
            self.magnitudeCorrectionPlot,
        )
