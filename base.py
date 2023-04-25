import dotmap


from PyQt6.QtWidgets import QMainWindow
from main_ui import Ui_MainWindow

from matplotlib import pyplot as plt

from photometry import LogCalibration
from matchers import Counselor


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
        self.param_widgets = [
            (self.dsb_x0, 'x0'), (self.dsb_y0, 'y0'), (self.dsb_a0, 'a0'), (self.dsb_A, 'A'), (self.dsb_F, 'F'),
            (self.dsb_V, 'V'), (self.dsb_S, 'S'), (self.dsb_D, 'D'), (self.dsb_P, 'P'), (self.dsb_Q, 'Q'),
            (self.dsb_eps, 'eps'), (self.dsb_E, 'E')
        ]

        self.settings = dotmap.DotMap(dict(
            resolution=dict(left=-1, bottom=-1, right=1, top=1)
        ))

        self.calibration = LogCalibration(4000)

    @property
    def paired(self) -> bool:
        return isinstance(self.matcher, Counselor)
