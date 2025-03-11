import dotmap
import numpy as np

from matplotlib import pyplot as plt
from collections import OrderedDict

from PyQt6.QtWidgets import QMainWindow
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

    def showErrors(self) -> None:
        rms_error = self.matcher.rms_error(self.position_errors)
        max_error = self.matcher.max_error(self.position_errors)
        self.lb_rms_error.setText(f'{np.degrees(rms_error):.6f}°')
        self.lb_max_error.setText(f'{np.degrees(max_error):.6f}°')

        errors = self.matcher.distance_sky(self.projection, mask_catalogue=True, mask_sensor=True)
        dot_to_nearest_star = np.min(errors, axis=1, initial=np.inf)
        star_to_nearest_dot = np.min(errors, axis=0, initial=np.inf)

        limit = np.radians(self.dsb_sensor_limit_dist.value())
        outside_limit = dot_to_nearest_star[dot_to_nearest_star > limit].size
        self.lb_sensor_outside.setText(f'{outside_limit}')

        limit = np.radians(self.dsb_catalogue_limit_dist.value())
        outside_limit = star_to_nearest_dot[star_to_nearest_dot > limit].size
        self.lb_catalogue_outside.setText(f'{outside_limit}')
