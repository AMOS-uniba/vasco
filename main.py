#!/usr/bin/env python

import sys
import yaml
import datetime
import zoneinfo
import numpy as np
import dotmap

from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QFileDialog

from astropy import units as u
from astropy.coordinates import EarthLocation

import matplotlib as mpl

from matchers import Matchmaker, Counselor
from projections import BorovickaProjection
from plotting import MainWindowPlots

from amos import AMOS, Station

mpl.use('Qt5Agg')

COUNT = 100


class MainWindow(MainWindowPlots):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.loadYAML('data/M20230313_131755_KY_00006.yaml')  # temporary
        self.importConstants('calibrations/KY.yaml')  # temporary
        self.maskCatalogueDistant()
        self.pair()

        self.connectSignalSlots()
        self.onParametersChanged()

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
        self.matcher.update_position_smoother(self.projection)

    def updateProjection(self):
        self.projection = BorovickaProjection(*self.getConstantsTuple())

    def onParametersChanged(self):
        print("Parameters changed")
        self.updateProjection()

        self.positionSkyPlot.invalidate_dots()
        self.positionSkyPlot.invalidate_meteor()
        self.magnitudeSkyPlot.invalidate_dots()
        self.magnitudeSkyPlot.invalidate_meteor()
        self.positionErrorPlot.invalidate()
        self.magnitudeErrorPlot.invalidate()
        self.positionCorrectionPlot.invalidate()
        self.magnitudeCorrectionPlot.invalidate()
        self.matcher.update_position_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()

    def onLocationTimeChanged(self):
        self.updateMatcher()
        self.positionSkyPlot.invalidate_stars()
        self.magnitudeSkyPlot.invalidate_stars()
        self.positionErrorPlot.invalidate()
        self.magnitudeErrorPlot.invalidate()
        self.matcher.update_position_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=self.dsb_bandwidth.value())
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

    def onBandwidthChanged(self):
        self.matcher.update_position_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=self.dsb_bandwidth.value())
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

    def computePositionErrors(self):
        self.position_errors = self.matcher.position_errors(self.projection, masked=True)

    def computeMagnitudeErrors(self):
        self.magnitude_errors = self.matcher.magnitude_errors(self.projection, self.calibration, masked=True)

    def exportFile(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export constants to file", ".", "YAML files (*.yaml)")
        if filename is not None:
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
        filename, _ = QFileDialog.getOpenFileName(self, "Load Kvant YAML file", "data",
                                                  "YAML files (*.yml *.yaml)")
        if filename == '':
            print("No file provided, loading aborted")
        else:
            self.loadYAML(filename)

        self.cb_stations.setCurrentIndex(0)
        self.onLocationTimeChanged()
        self.onParametersChanged()
        self.sensorPlot.invalidate()
        self.updatePlots()

    def loadYAML(self, file):
        data = dotmap.DotMap(yaml.safe_load(open(file, 'r')))
        self.setLocation(data.Latitude, data.Longitude)
        self.updateLocation()
        self.setTime(datetime.datetime
                     .strptime(data.EventStartTime, "%Y-%m-%d %H:%M:%S.%f")
                     .replace(tzinfo=zoneinfo.ZoneInfo('UTC')))
        self.updateTime()

        self.matcher = Matchmaker(self.location, self.time)
        self.matcher.sensor_data.load(data)
        self.matcher.load_catalogue('catalogue/HYG30.tsv')

    def importFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Import constants from file", ".",
                                                  "YAML files (*.yml *.yaml)")
        self.importConstants(filename)
        self.onParametersChanged()

    def importConstants(self, filename):
        try:
            with open(filename, 'r') as file:
                try:
                    data = dotmap.DotMap(yaml.safe_load(file))
                    self.blockParameterSignals(True)
                    for widget, param in self.param_widgets:
                        widget.setValue(data.params[param])
                    self.blockParameterSignals(False)

                    self.updateProjection()
                except yaml.YAMLError as exc:
                    print(f"Could not open file {filename}: {exc}")
        except FileNotFoundError as exc:
            print(f"Could not import constants: {exc}")

    def blockParameterSignals(self, block):
        for widget, param in self.param_widgets:
            widget.blockSignals(block)

    def getConstantsTuple(self):
        return (
            self.dsb_x0.value(),
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
            x0=self.getConstantsTuple(),
            maxiter=self.sb_maxiter.value()
        )

        x0, y0, a0, A, F, V, S, D, P, Q, e, E = tuple(result.x)
        self.blockParameterSignals(True)
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
        self.blockParameterSignals(False)

        self.w_input.setEnabled(True)
        self.w_input.repaint()
        self.onParametersChanged()

    def maskSensor(self):
        errors = self.matcher.position_errors(self.projection, masked=False)
        self.matcher.mask_sensor_data(errors > np.radians(self.dsb_error_limit.value()))
        print(f"Culled the dots to {self.dsb_error_limit.value()}°: "
              f"{self.matcher.sensor_data.stars.count_valid} are valid")
        self.onParametersChanged()
        self.showCounts()

    def maskCatalogueDistant(self):
        errors = self.matcher.position_errors_inverse(self.projection, masked=False)
        self.matcher.mask_catalogue(errors > np.radians(self.dsb_distance_limit.value()))
        print(f"Culled the catalogue to {self.dsb_distance_limit.value()}°: "
              f"{self.matcher.catalogue.count_valid} stars used")
        self.positionSkyPlot.invalidate_stars()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()
        self.showCounts()

    def maskCatalogueFaint(self):
        self.matcher.mask_catalogue(self.matcher.catalogue.vmag > self.dsb_magnitude_limit.value())
        print(f"Culled the catalogue to {self.dsb_magnitude_limit.value()}m: "
              f"{self.matcher.catalogue.count_valid} stars used")
        self.positionSkyPlot.invalidate_stars()

        self.computePositionErrors()
        self.computeMagnitudeErrors()
        self.updatePlots()
        self.showCounts()

    def resetValid(self):
        self.matcher.reset_mask()
        self.onParametersChanged()
        self.showCounts()

    @property
    def paired(self) -> bool:
        return isinstance(self.matcher, Counselor)

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

    def showErrors(self) -> None:
        avg_error = self.matcher.avg_error(self.position_errors)
        max_error = self.matcher.max_error(self.position_errors)
        self.lb_avg_error.setText(f'{np.degrees(avg_error):.6f}°')
        self.lb_max_error.setText(f'{np.degrees(max_error):.6f}°')
        self.lb_total_stars.setText(f'{self.matcher.catalogue.count}')
        outside_limit = self.position_errors[self.position_errors > np.radians(self.dsb_error_limit.value())].size
        self.lb_outside_limit.setText(f'{outside_limit}')

    def correctMeteor(self):
        if self.paired:
            filename, _ = QFileDialog.getSaveFileName(self, "Export corrected meteor to file", ".", "XML files (*.xml)")
            if filename is not None and filename != '':
                with open(filename, 'w') as file:
                    file.write(f"""
<?xml version="1.0" encoding="UTF-8" ?>
<ufoanalyzer_record version ="200"
    clip_name="M20120922_225744_ago_" o="1" y="2012" mo="9"
    d="22" h="22" m="57" s="44.0"
    tz="0" tme="0" lid="ago" sid="kvant"
    lng="{self.dsb_lon.value()}" lat="{self.dsb_lat.value()}" alt="0"
    cx="{self.matcher.sensor_data.rect.xmax} cy="{self.matcher.sensor_data.rect.ymax}" fps="15" interlaced="0" bbf="0"
    frames="{self.matcher.sensor_data.meteor.count}" head="19" tail="0" drop="-1"
    dlev="0" dsize="0" sipos="0" sisize="0"
    trig="0" observer="ago" cam="" lens=""
    cap="" u2="0" ua="0" memo=""
    az="0" ev="0" rot="0" vx="0"
    yx="0" dx="0" dy="0" k4="0"
    k3="0" k2="0" atc="0" BVF="0"
    maxLev="0" maxMag="0" minLev="0" mimMag="0"
    dl="0" leap="0" pixs="0" rstar="0.0283990702993807"
    ddega="0.03276" ddegm="0" errm="0" Lmrgn="0"
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
            sigma="0.03276"
            sigma.azi="0.0283990702993807"
            sigma.zen="0.0354915982362712"
            A0="-0.05584542"
            X0="-0.00874173"
            Y0="0.01862264"
            V="0.64574381"
            S="-0.46404608"
            D="0.25"
            EPS="0.00206778"
            E="3.36956865"
            A="0.0015977"
            F0="0.75477673"
            P="0.150045"
            Q="0.066"
            C="1"
            CH1="690"
            CH2="0.00494625"
            CH3="535"
            CH4="0.004992"
            magA="7.34131408453742"
            magB="1.50852603934261"
            magR2="0.548439901974541"
            magS="0.399746731883133"
            usingPrecession="True">""")
                    file.write(self.matcher.print_meteor(self.projection, self.calibration))
                    file.write("""
        </ua2_object>
    </ua2_objects>
</ufoanalyzer_record>""")

    @property
    def grid_resolution(self):
        return self.sb_resolution.value()

    def pair(self):
        self.matcher = self.matcher.pair(self.projection)
        self.matcher.update_position_smoother(self.projection, bandwidth=self.dsb_bandwidth.value())
        self.matcher.update_magnitude_smoother(self.projection, self.calibration, bandwidth=self.dsb_bandwidth.value())

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


app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()

app.exec()
