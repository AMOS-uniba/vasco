from typing import Callable

import dotmap
import numpy as np

from matplotlib import pyplot as plt
from collections import OrderedDict

from PyQt6.QtWidgets import QMainWindow, QDoubleSpinBox, QLabel
from main_ui import Ui_MainWindow


from photometry import LogCalibration
from matchers import Counsellor


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

        self.settings = dotmap.DotMap(dict(
            resolution=dict(left=-1, bottom=-1, right=1, top=1)
        ))

        self.calibration = LogCalibration(4000)

    @property
    def paired(self) -> bool:
        return isinstance(self.matcher, Counsellor)

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

        errors = self.matcher.distance_sky(self.projection, mask_catalogue=True, mask_sensor=True, paired=self.paired)

        self._update_maskable_count(self.dsb_sensor_limit_dist, self.lb_sensor_dist,
                                    np.min(errors, axis=1, initial=np.inf),
                                    func=lambda x: np.radians(x))
        self._update_maskable_count(self.dsb_sensor_limit_alt, self.lb_sensor_alt,
                                    self.matcher.sensor_data.stars.project(self.projection, masked=True)[..., 0],
                                    func=lambda x: np.radians(90 - x))

        self._update_maskable_count(self.dsb_catalogue_limit_dist, self.lb_catalogue_dist,
                                    np.min(errors, axis=0, initial=np.inf),
                                    func=lambda x: np.radians(x))
        self._update_maskable_count(self.dsb_catalogue_limit_alt, self.lb_catalogue_alt,
                                    self.matcher.altaz(masked=True)[..., 0],
                                    func=lambda x: np.radians(x),
                                    invert_mask=True)
        self._update_maskable_count(self.dsb_catalogue_limit_mag, self.lb_catalogue_mag,
                                    self.matcher.vmag(masked=True))
