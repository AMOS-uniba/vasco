import logging

import numpy as np

from typing import Callable

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

        self.sensor_plot = SensorPlot(self.tab_sensor_plot)
        self.position_sky_plot = PositionSkyPlot(self.tab_sky_positions)
        self.magnitude_sky_plot = MagnitudeSkyPlot(self.tab_sky_magnitudes)
        self.position_error_plot = PositionErrorPlot(self.tab_errors_positions)
        self.magnitude_error_plot = MagnitudeErrorPlot(self.tab_errors_magnitudes)
        self.position_correction_plot = PositionCorrectionPlot(self.tab_correction_positions)
        self.magnitude_correction_plot = MagnitudeCorrectionPlot(self.tab_correction_magnitudes)

        # List of links: when the i-th tab is active and update_plots is called,
        # iterate over the i-th element here anmd for every pair (condition, action)
        # do action if condition is true
        self.updateable: [(Callable, Callable)] = [
            [
                (self.sensor_plot.valid, self.plot_sensor_data),
            ],
            [
                (False, self.update_stars_table),
            ],
            [
                (self.position_sky_plot.valid_dots, self.plot_observed_stars_positions),
                (self.position_sky_plot.valid_stars, self.plot_catalogue_stars_positions),
            ],
            [
                (self.magnitude_sky_plot.valid_dots, self.plot_observed_stars_magnitudes),
                (self.magnitude_sky_plot.valid_stars, self.plot_catalogue_stars_magnitudes),
            ],
            [
                (self.position_error_plot.valid_dots, self.plot_position_errors_dots),
                (self.position_error_plot.valid_meteor, self.plot_position_errors_meteor),
            ],
            [
                (self.magnitude_error_plot.valid_dots, self.plot_magnitude_errors_dots),
                (self.magnitude_error_plot.valid_meteor, self.plot_magnitude_errors_meteor),
            ],
            [
                (self.position_correction_plot.valid_dots, self.plot_position_correction_errors),
                (self.position_correction_plot.valid_meteor, self.plot_position_correction_meteor),
                (self.position_correction_plot.valid_grid, self.plot_position_correction_grid),
            ],
            [
                (self.magnitude_correction_plot.valid_dots, self.plot_magnitude_correction_errors),
                (self.magnitude_correction_plot.valid_meteor, self.plot_magnitude_correction_meteor),
                (self.magnitude_correction_plot.valid_grid, self.plot_magnitude_correction_grid),
            ],
            [
                (False, self.update_meteor_table),
            ],
            [
                (False, self.update_catalogue_table),
            ],
        ]

    def update_plots(self):
        """
        Iterate over active plots and update those that are no longer valid
        """
        self.show_errors()

        for valid, function in self.updateable[self.tw_charts.currentIndex()]:
            if not valid:
                function()

# Methods for plotting sensor data

    def plot_sensor_data(self):
        self.sensor_plot.update(self.matcher.sensor_data)

# Methods for plotting sky charts

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
        self._plot_observed_stars(self.position_sky_plot, self.position_errors,
                                  limit=np.radians(self.dsb_sensor_limit_dist.value()))

    def plot_observed_stars_magnitudes(self):
        log.debug(f"Plotting dot magnitudes")
        self._plot_observed_stars(self.magnitude_sky_plot, self.magnitude_errors,
                                  limit=np.radians(self.dsb_sensor_limit_dist.value()))

    def _plot_catalogue_stars(self, plot):
        plot.update_stars(
            self.matcher.catalogue_altaz_np(masked=True),
            self.matcher.catalogue_vmag(masked=True)
        )

    def plot_catalogue_stars_positions(self):
        log.debug(f"Plotting star positions")
        self._plot_catalogue_stars(self.position_sky_plot)

    def plot_catalogue_stars_magnitudes(self):
        log.debug(f"Plotting star magnitudes")
        self._plot_catalogue_stars(self.magnitude_sky_plot)

    """ Methods for plotting error charts """

    def _plot_errors_dots(self, plot, errors):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.stars.intensities(True)
        plot.update_dots(positions, magnitudes, errors, limit=self.dsb_sensor_limit_dist.value())

    def plot_position_errors_dots(self):
        log.debug(f"Plotting position errors")
        self._plot_errors_dots(self.position_error_plot, self.position_errors)

    def plot_magnitude_errors_dots(self):
        log.debug(f"Plotting magnitude errors")
        self._plot_errors_dots(self.magnitude_error_plot, self.magnitude_errors)

    def _plot_errors_meteor(self, plot, errors):
        positions = self.matcher.sensor_data.meteor.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.meteor.intensities(True)
        plot.update_meteor(positions, magnitudes, errors, limit=0.1)

    def plot_position_errors_meteor(self):
        xy = self.matcher.correction_meteor_xy(self.projection)
        correction = np.degrees(np.sqrt(xy[..., 0]**2 + xy[..., 1]**2))
        self._plot_errors_meteor(self.position_error_plot, correction)

    def plot_magnitude_errors_meteor(self):
        self._plot_errors_meteor(self.magnitude_error_plot, self.matcher.correction_meteor_mag(self.projection))

    """ Methods for updating correction plots """

    def _plot_correction_errors(self, plot: BaseCorrectionPlot) -> None:
        if self.cb_show_errors.isChecked():
            log.debug(f"Plotting {plot.intent} for {plot.target}")
            plot.update_dots(
                self.matcher.catalogue_altaz_paired()[self.matcher.sensor_data.stars.mask],
                self.matcher.sensor_data.stars.project(self.projection, masked=True),
                self.matcher.catalogue_vmag_paired()[self.matcher.sensor_data.stars.mask],
                self.matcher.sensor_data.stars.calibrate(self.calibration, masked=True),
                limit=np.radians(self.dsb_sensor_limit_dist.value()),
                scale=1000 / self.sb_arrow_scale.value(),
            )
        else:
            plot.clear_errors()

    def plot_position_correction_errors(self) -> None:
        self._plot_correction_errors(self.position_correction_plot)

    def plot_magnitude_correction_errors(self) -> None:
        self._plot_correction_errors(self.magnitude_correction_plot)

    def _plot_correction_meteor(self, plot) -> None:
        plot.update_meteor(
            self.matcher.sensor_data.meteor.project(self.projection, masked=True),
            self.matcher.correction_meteor_xy(self.projection),
            self.matcher.sensor_data.meteor.calibrate(self.calibration, masked=True),
            self.matcher.correction_meteor_mag(self.projection),
            scale=1 / self.sb_arrow_scale.value(),
        )

    def plot_position_correction_meteor(self) -> None:
        self._plot_correction_meteor(self.position_correction_plot)

    def plot_magnitude_correction_meteor(self) -> None:
        self._plot_correction_meteor(self.magnitude_correction_plot)

    def _plot_correction_grid(self, plot, grid, *, masked: bool, **kwargs):
        if self.cb_show_grid.isChecked():
            xx, yy = unit_grid(self.grid_resolution, masked=masked)
            plot.update_grid(xx, yy, grid(resolution=self.grid_resolution), **kwargs)
        else:
            plot.clear_grid()

    def plot_position_correction_grid(self):
        self._plot_correction_grid(self.position_correction_plot, self.matcher.position_grid, masked=True)

    def plot_magnitude_correction_grid(self):
        self._plot_correction_grid(self.magnitude_correction_plot, self.matcher.magnitude_grid, masked=False,
                                   interpolation=self.cb_interpolation.currentText())
