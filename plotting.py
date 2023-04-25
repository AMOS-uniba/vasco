import numpy as np

from typing import Callable

from PyQt6.QtWidgets import QStackedWidget


from plots import SensorPlot
from plots.sky import PositionSkyPlot, MagnitudeSkyPlot
from plots.errors import PositionErrorPlot, MagnitudeErrorPlot
from plots.correction import BaseCorrectionPlot, PositionCorrectionPlot, MagnitudeCorrectionPlot
from utilities import unit_grid

from base import MainWindowBase


class MainWindowPlots(MainWindowBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sensorPlot = SensorPlot(self.tab_sensor)
        self.positionSkyPlot = PositionSkyPlot(self.tab_sky_positions)
        self.magnitudeSkyPlot = MagnitudeSkyPlot(self.tab_sky_magnitudes)
        self.positionErrorPlot = PositionErrorPlot(self.tab_errors_positions)
        self.magnitudeErrorPlot = MagnitudeErrorPlot(self.tab_errors_magnitudes)
        self.positionCorrectionPlot = PositionCorrectionPlot(self.tab_correction_positions_enabled)
        self.magnitudeCorrectionPlot = MagnitudeCorrectionPlot(self.tab_correction_magnitudes_enabled)

    def updatePlots(self):
        self.showErrors()
        links = [
            [
                (self.sensorPlot.valid, self.plotSensorData),
            ],
            [
                (self.positionSkyPlot.valid_dots, self.plotObservedStarsPositions),
                (self.positionSkyPlot.valid_stars, self.plotCatalogueStarsPositions),
            ],
            [
                (self.magnitudeSkyPlot.valid_dots, self.plotObservedStarsMagnitudes),
                (self.magnitudeSkyPlot.valid_stars, self.plotCatalogueStarsMagnitudes),
            ],
            [
                (self.positionErrorPlot.valid_dots, self.plotPositionErrorsDots),
                (self.positionErrorPlot.valid_meteor, self.plotPositionErrorsMeteor),
            ],
            [
                (self.magnitudeErrorPlot.valid_dots, self.plotMagnitudeErrorsDots),
                (self.magnitudeErrorPlot.valid_meteor, self.plotMagnitudeErrorsMeteor),
            ],
            [
                (self.positionCorrectionPlot.valid_dots, self.plotPositionCorrectionErrors),
                (self.positionCorrectionPlot.valid_meteor, self.plotPositionCorrectionMeteor),
                (self.positionCorrectionPlot.valid_grid, self.plotPositionCorrectionGrid),
            ],
            [
                (self.magnitudeCorrectionPlot.valid_dots, self.plotMagnitudeCorrectionErrors),
                (self.magnitudeCorrectionPlot.valid_meteor, self.plotMagnitudeCorrectionMeteor),
                (self.magnitudeCorrectionPlot.valid_grid, self.plotMagnitudeCorrectionGrid),
            ],
        ][self.tw_charts.currentIndex()]

        for valid, function in links:
            if not valid:
                function()

    """ Methods for plotting sensor data """

    def plotSensorData(self):
        self.sensorPlot.update(self.matcher.sensor_data)

    """ Methods for plotting sky charts """

    def _plotObservedStars(self, plot, errors, *, limit=None):
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

    def plotObservedStarsPositions(self):
        self._plotObservedStars(self.positionSkyPlot, self.position_errors,
                                limit=np.radians(self.dsb_error_limit.value()))

    def plotObservedStarsMagnitudes(self):
        self._plotObservedStars(self.magnitudeSkyPlot, self.magnitude_errors,
                                limit=np.radians(self.dsb_error_limit.value()))

    def _plotCatalogueStars(self, plot):
        plot.update_stars(
            self.matcher.catalogue.to_altaz_chart(self.location, self.time, masked=True),
            self.matcher.catalogue.vmag(masked=True)
        )

    def plotCatalogueStarsPositions(self):
        self._plotCatalogueStars(self.positionSkyPlot)

    def plotCatalogueStarsMagnitudes(self):
        self._plotCatalogueStars(self.magnitudeSkyPlot)

    """ Methods for plotting error charts """

    def _plotErrorsDots(self, plot, errors):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.stars.intensities(True)
        plot.update_dots(positions, magnitudes, errors, limit=np.radians(self.dsb_error_limit.value()))

    def plotPositionErrorsDots(self):
        self._plotErrorsDots(self.positionErrorPlot, self.position_errors)

    def plotMagnitudeErrorsDots(self):
        self._plotErrorsDots(self.magnitudeErrorPlot, self.magnitude_errors)

    def _plotErrorsMeteor(self, plot, errors):
        positions = self.matcher.sensor_data.meteor.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.meteor.intensities(True)
        plot.update_meteor(positions, magnitudes, errors, limit=0.1)

    def plotPositionErrorsMeteor(self):
        if self.paired:
            xy = self.matcher.correction_meteor_xy(self.projection)
            correction = np.degrees(np.sqrt(xy[..., 0]**2 + xy[..., 1]**2))
            self._plotErrorsMeteor(self.positionErrorPlot, correction)

    def plotMagnitudeErrorsMeteor(self):
        if self.paired:
            self._plotErrorsMeteor(self.magnitudeErrorPlot, self.matcher.correction_meteor_mag(self.projection))

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
            print(f"Plotting {plot.intent} for {plot.target}")
            plot.update_dots(
                self.matcher.catalogue.altaz(self.location, self.time, masked=True),
                self.matcher.sensor_data.stars.project(self.projection, masked=True),
                self.matcher.catalogue.vmag(masked=True),
                self.matcher.sensor_data.stars.calibrate(self.calibration, masked=True),
                limit=np.radians(self.dsb_error_limit.value()),
                scale=1 / self.sb_arrow_scale.value(),
            )
        else:
            plot.clear_errors()

    def plotPositionCorrectionErrors(self) -> None:
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

    def plotPositionCorrectionGrid(self):
        self._switch_tabs(
            self.tabs_positions,
            lambda plot: self._plotCorrectionGrid(plot, self.matcher.position_grid, masked=True),
            self.positionCorrectionPlot,
        )

    def plotMagnitudeCorrectionGrid(self):
        self._switch_tabs(
            self.tabs_magnitudes,
            lambda plot: self._plotCorrectionGrid(plot, self.matcher.magnitude_grid, masked=False,
                                                  interpolation=self.cb_interpolation.currentText()),
            self.magnitudeCorrectionPlot,
        )
