from typing import Callable

import dotmap
import numpy as np

from matplotlib import pyplot as plt
from collections import OrderedDict

from PyQt6.QtWidgets import QMainWindow, QDoubleSpinBox, QLabel
from main_ui import Ui_MainWindow

from photometry import LogCalibration
from models import Matcher


class MainWindowBase(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        plt.style.use('dark_background')
        self.position_errors = None
        self.magnitude_errors = None
        self.location = None
        self.time = None
        self.projection = None
        self.calibration = None
        self.matcher = None

        self.setupUi(self)
        self.param_widgets = OrderedDict(
            x0=self.pw_x0,
            y0=self.pw_y0,
            a0=self.pw_a0,
            A=self.pw_A,
            F=self.pw_F,
            V=self.pw_V,
            S=self.pw_S,
            D=self.pw_D,
            P=self.pw_P,
            Q=self.pw_Q,
            epsilon=self.pw_epsilon,
            E=self.pw_E,
        )

        self.location_time_widgets = OrderedDict(
            lat=self.dsb_lat,
            lon=self.dsb_lon,
            alt=self.dsb_alt,
            time=self.dt_time,
        )

        self.pixel_scale_widgets = OrderedDict(
            xs=self.dsb_xs,
            ys=self.dsb_ys,
        )

        self.settings = dotmap.DotMap(dict(
            resolution=dict(left=-1, bottom=-1, right=1, top=1)
        ))

        self.calibration = LogCalibration(4000)

    def _update_maskable_count(self,
                               dsb_in: QDoubleSpinBox,
                               lb_out: QLabel,
                               values: np.ndarray[float],
                               *,
                               func: Callable[[float], float] = lambda x: x,
                               invert_mask: bool = False):
        """
        Helper function:
        - take value from QDoubleSpinBox `dsb_in`,
        - filter `values` that are larger (or smaller, if `invert` is True,
        - output their count to `lb_out`.
        """
        limit = func(dsb_in.value())
        mask = values < limit if invert_mask else values > limit
        outside_limit = values[mask].size
        lb_out.setText(f'{-outside_limit}')

    def show_errors(self) -> None:
        rms_error = self.matcher.rms_error(self.position_errors)
        max_error = self.matcher.max_error(self.position_errors)
        self.lb_rms_error.setText(f'{np.degrees(rms_error):.6f}°')
        self.lb_max_error.setText(f'{np.degrees(max_error):.6f}°')

        errors = self.matcher.distance_sky_full()

        sensor_dist_errors = np.min(errors, axis=1, initial=np.inf)[self.matcher.sensor_data.stars.mask]
        catalogue_dist_errors = np.min(errors, axis=0, initial=np.inf)[self.matcher.catalogue.mask]

        self._update_maskable_count(self.dsb_sensor_limit_dist, self.lb_sensor_dist,
                                    sensor_dist_errors,
                                    func=lambda x: np.radians(x))
        self._update_maskable_count(self.dsb_sensor_limit_alt, self.lb_sensor_alt,
                                    self.matcher.sensor_data.stars.project(self.projection, masked=True, flip_theta=True)[..., 0],
                                    func=lambda x: np.radians(x),
                                    invert_mask=True)

        self._update_maskable_count(self.dsb_catalogue_limit_dist, self.lb_catalogue_dist,
                                    catalogue_dist_errors,
                                    func=lambda x: np.radians(x))
        self._update_maskable_count(self.dsb_catalogue_limit_alt, self.lb_catalogue_alt,
                                    self.matcher.catalogue_altaz_np(masked=True)[..., 0],
                                    func=lambda x: np.radians(x),
                                    invert_mask=True)
        self._update_maskable_count(self.dsb_catalogue_limit_mag, self.lb_catalogue_mag,
                                    self.matcher.catalogue_vmag(masked=True))
