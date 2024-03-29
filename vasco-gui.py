#!/usr/bin/env python
import logging
import sys
import argparse
import logger


from PyQt6.QtWidgets import QApplication
from mainwindow import MainWindow


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('-c', '--catalogue', type=argparse.FileType('r'))
parser.add_argument('-s', '--sighting', type=argparse.FileType('r'))
parser.add_argument('-p', '--projection', type=argparse.FileType('r'))
args = parser.parse_args()

log = logger.setupLog('vasco')
log.setLevel(logging.DEBUG if args.debug else logging.INFO)

app = QApplication(sys.argv)
window = MainWindow(args)
window.showMaximized()

app.exec()
