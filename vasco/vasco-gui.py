#!/usr/bin/env python
import logging
import sys
import argparse
import logger


from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from mainwindow import MainWindow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-c', '--catalogue', type=argparse.FileType('r'))
    parser.add_argument('-s', '--sighting', type=argparse.FileType('r'))
    parser.add_argument('-p', '--projection', type=argparse.FileType('r'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log = logger.setupLog('vasco')
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)

    log.debug("vasco is starting")

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/vasco.png'))

    window = MainWindow(args)
    window.setWindowIcon(QIcon('assets/vasco.png'))
    window.show()
    #window.showMaximized()

    app.exec()
