#!/usr/bin/env python

import sys

from PyQt6.QtWidgets import QApplication
from mainwindow import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()

app.exec()
