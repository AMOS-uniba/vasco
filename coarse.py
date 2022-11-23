#!/usr/bin/env python

import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt

from typing import Tuple, Type, Optional

import PyQt6
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QMainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from shifters import OpticalAxisShifter, EllipticShifter
from transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer
from projections import Projection, BorovickaProjection


mpl.use('Qt5Agg')

COUNT = 100


class Fitter():
    def __init__(self):
        pass

    def __call__(self, xy: Tuple[np.ndarray, np.ndarray], za: Tuple[np.ndarray, np.ndarray], cls: Type[Projection], *, params: Optional[dict]=None) -> Projection:
        """
            xy      a 2-tuple of x and y coordinates on the sensor
            za      a 2-tuple of z and a coordinates in the sky catalogue
            cls     a subclass of Projection that is used to transform xy onto za
            Returns an instance of cls with parameters set to values that result in minimal deviation
        """
        return cls(params)


class Vascop():
    """ Virtual All-Sky CorrectOr Plate """
    def __init__(self):
        self.argparser = argparse.ArgumentParser("Virtual all-sky corrector plate")
        self.argparser.add_argument('infile', type=argparse.FileType('r'), help="input file")
        self.argparser.add_argument('outdir', action=argparser.WriteableDir, help="output directory")
        self.argparser.add_argument('method', type=str, choices=Corrector.METHODS)
        self.args = self.argparser.parse_args()
        self.outdir = Path(self.args.outdir)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual All-Sky CorrectOr Plate")

        self.spinBox = QtWidgets.QDoubleSpinBox()
        self.spinBox.setMinimum(-1)
        self.spinBox.setSingleStep(0.001)
        self.spinBox.setMaximum(1)

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots(1, 2)
        self.figure.tight_layout()
        self.plot()

        self.entire = QtWidgets.QWidget()
        self.globalLayout = QtWidgets.QGridLayout()
        self.entire.setLayout(self.globalLayout)
        self.setCentralWidget(self.entire)

        self.control = QtWidgets.QGroupBox(self)
        self.control.setTitle("Controls")
        self.controlLayout = QtWidgets.QHBoxLayout()

        self.controlLayout.addWidget(self.build_group_box(self.control, "Sensor", params=[
            {'title': 'x0', 'kwargs': dict(minimum=-1000, maximum=1000, value=0, step=0.1)},
            {'title': 'y0', 'kwargs': dict(minimum=-1000, maximum=1000, value=0, step=0.1)},
            {'title': 'a0', 'kwargs': dict(minimum=0, maximum=2 * np.pi, value=0, step=0.001)},
        ]))

        self.controlLayout.addWidget(self.build_group_box(self.control, "Elliptic distortion", params=[
            {'title': 'A', 'kwargs': dict(minimum=0, maximum=1, value=0, step=0.001)},
            {'title': 'F', 'kwargs': dict(minimum=0, maximum=2 * np.pi, value=0, step=0.001)},
        ]))

        self.controlLayout.addWidget(self.build_group_box(self.control, "Lens", params=[
            {'title': 'V', 'kwargs': dict(minimum=0, maximum=2, value=1, step=0.001)},
            {'title': 'S', 'kwargs': dict(minimum=-1, maximum=1, value=0, step=0.001)},
            {'title': 'D', 'kwargs': dict(minimum=0, maximum=5, value=0, step=0.001)},
            {'title': 'P', 'kwargs': dict(minimum=-1, maximum=1, value=0, step=0.001)},
            {'title': 'Q', 'kwargs': dict(minimum=0, maximum=5, value=0, step=0.001)},
        ]))

        self.controlLayout.addWidget(self.build_group_box(self.control, "Zenith", params=[
            {'title': 'epsilon', 'kwargs': dict(minimum=0, maximum=np.pi, value=0, step=0.001)},
            {'title': 'E', 'kwargs': dict(minimum=0, maximum=2 * np.pi, value=0, step=0.001)},
        ]))

        self.control.setLayout(self.controlLayout)
        self.globalLayout.addWidget(self.control)

    def build_group_box(self, parent, title, params):
        groupbox = QtWidgets.QGroupBox()
        groupbox.setTitle(title)
        layout = QtWidgets.QHBoxLayout(groupbox)

        for param in params:
            layout.addWidget(self.build_parameter_box(groupbox, param['title'], **param['kwargs']), len(param['kwargs']))

        groupbox.setLayout(layout)

        return groupbox


    def build_parameter_box(self, parent, title, **kwargs):
        box = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(box)

        label = QtWidgets.QLabel(box)
        label.setText(title)
        spinbox = QtWidgets.QDoubleSpinBox(box)
        spinbox.setMinimum(kwargs.get('minimum', -100))
        spinbox.setMaximum(kwargs.get('maximum', 100))
        spinbox.setValue(kwargs.get('value', 0))
        spinbox.setSingleStep(kwargs.get('step', 0.01))
        spinbox.setDecimals(kwargs.get('decimals', 6))

        layout.addWidget(label)
        layout.addWidget(spinbox)
        box.setLayout(layout)

        return box


    def plot(self):
        x = np.random.normal(0, 0.3, size=COUNT)
        y = np.random.normal(0, 0.3, size=COUNT)
        self.ax[0].scatter(x, y)



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
