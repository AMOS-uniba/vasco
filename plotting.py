import numpy as np

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

    @staticmethod
    def _updateIfInvalid(valid, function):
        if not valid:
            function()

    def updatePlots(self):
        self.showErrors()
        self.correctMeteor()
        index = self.tw_charts.currentIndex()

        if index == 0:
            self._updateIfInvalid(self.sensorPlot.valid, self.plotSensorData)
        elif index == 1:
            self._updateIfInvalid(self.positionSkyPlot.valid_dots, self.plotObservedStarsPositions)
            self._updateIfInvalid(self.positionSkyPlot.valid_stars, self.plotCatalogueStarsPositions)
        elif index == 2:
            self._updateIfInvalid(self.magnitudeSkyPlot.valid_dots, self.plotObservedStarsMagnitudes)
            self._updateIfInvalid(self.magnitudeSkyPlot.valid_stars, self.plotCatalogueStarsMagnitudes)
        elif index == 3:
            self._updateIfInvalid(self.positionErrorPlot.valid, self.plotPositionErrors)
        elif index == 4:
            self._updateIfInvalid(self.positionErrorPlot.valid, self.plotMagnitudeErrors)
        elif index == 5:
            self._updateIfInvalid(self.positionCorrectionPlot.valid_dots, self.plotPositionCorrectionErrors)
            self._updateIfInvalid(self.positionCorrectionPlot.valid_meteor, self.plotPositionCorrectionMeteor)
            self._updateIfInvalid(self.positionCorrectionPlot.valid_grid, self.plotPositionCorrectionGrid)
        elif index == 6:
            self._updateIfInvalid(self.magnitudeCorrectionPlot.valid_dots, self.plotMagnitudeCorrectionErrors)
            self._updateIfInvalid(self.magnitudeCorrectionPlot.valid_meteor, self.plotMagnitudeCorrectionMeteor)
            self._updateIfInvalid(self.magnitudeCorrectionPlot.valid_grid, self.plotMagnitudeCorrectionGrid)

    def plotSensorData(self):
        self.sensorPlot.update(self.matcher.sensor_data)

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

    def _plotErrors(self, plot, errors):
        positions = self.matcher.sensor_data.stars.project(self.projection, masked=True)
        magnitudes = self.matcher.sensor_data.stars.intensities(True)
        plot.update(positions, magnitudes, errors, limit=np.radians(self.dsb_error_limit.value()))

    def plotPositionErrors(self):
        self._plotErrors(self.positionErrorPlot, self.position_errors)

    def plotMagnitudeErrors(self):
        self._plotErrors(self.magnitudeErrorPlot, self.magnitude_errors)

    """ Methods for updating correction plots """

    def _plotCorrectionErrors(self, tabs: QStackedWidget, plot: BaseCorrectionPlot, *, intent: str) -> None:
        # Do nothing if working in unpaired mode
        if self.paired:
            tabs.setCurrentIndex(1)
            if self.cb_show_errors.isChecked():
                print(f"Plotting correction errors for {intent}")
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
        else:
            tabs.setCurrentIndex(0)

    def plotPositionCorrectionErrors(self):
        self._plotCorrectionErrors(self.tabs_positions, self.positionCorrectionPlot, intent="star positions")

    def plotMagnitudeCorrectionErrors(self):
        self._plotCorrectionErrors(self.tabs_magnitudes, self.magnitudeCorrectionPlot, intent="star magnitudes")

    def _plotCorrectionMeteor(self, tabs, plot, *, intent: str) -> None:
        if self.paired:
            print(f"Plotting correction for {intent}")
            tabs.setCurrentIndex(1)
            plot.update_meteor(
                self.matcher.sensor_data.meteor.project(self.projection, masked=True),
                self.matcher.correction_meteor_xy(self.projection),
                self.matcher.sensor_data.meteor.calibrate(self.calibration, masked=True),
                self.matcher.correction_meteor_mag(self.projection),
                scale=1 / self.sb_arrow_scale.value(),
            )
        else:
            tabs.setCurrentIndex(0)

    def plotPositionCorrectionMeteor(self) -> None:
        self._plotCorrectionMeteor(self.tabs_positions, self.positionCorrectionPlot, intent="star positions")

    def plotMagnitudeCorrectionMeteor(self) -> None:
        self._plotCorrectionMeteor(self.tabs_magnitudes, self.magnitudeCorrectionPlot, intent="star magnitudes")

    def switch_tabs(self, tabs, func, arg, *, message: str, intent: str) -> None:
        if self.paired:
            tabs.setCurrentIndex(1)
            print(f"{message} for {intent}: resolution {self.sb_resolution.value()}, ")
            func(arg)
        else:
            tabs.setCurrentIndex(0)

    def _plotCorrectionGrid(self, plot, grid, *, masked: bool, **kwargs):
        if self.cb_show_grid.isChecked():
            xx, yy = unit_grid(self.grid_resolution, masked=masked)
            plot.update_grid(xx, yy, grid(resolution=self.grid_resolution), **kwargs)
        else:
            plot.clear_grid()

    def plotPositionCorrectionGrid(self):
        self.switch_tabs(
            self.tabs_positions,
            lambda x: self._plotCorrectionGrid(x, self.matcher.position_grid, masked=True),
            self.positionCorrectionPlot,
            message="Plotting correction grid",
            intent="star positions",
        )

    def plotMagnitudeCorrectionGrid(self):
        self.switch_tabs(
            self.tabs_magnitudes,
            lambda x: self._plotCorrectionGrid(x, self.matcher.magnitude_grid,
                                               masked=False,
                                               interpolation=self.cb_interpolation.currentText()),
            self.magnitudeCorrectionPlot,
            message="Plotting correction grid",
            intent="star magnitudes",
        )
