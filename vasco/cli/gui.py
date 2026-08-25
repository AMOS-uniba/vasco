#!/usr/bin/env python
"""
The window. Was vasco/vasco-gui.py, which could only be run from inside vasco/vasco/.
"""
import argparse
import logging
import sys

from vasco import logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='vasco-gui')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-c', '--catalogue', type=argparse.FileType('r'))
    parser.add_argument('-s', '--sighting', type=argparse.FileType('r'))
    parser.add_argument('-p', '--projection', type=argparse.FileType('r'))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = logger.setupLog('vasco')
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    log.debug("vasco is starting")

    # Imported here and not at the top: PyQt6 is the `gui` extra, and this module is in the same
    # package as vasco-fit, which runs where Qt is not installed. Importing it at the top would
    # make `vasco-fit --help` depend on a window toolkit.
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    from vasco.mainwindow import MainWindow

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('assets/vasco.png'))

    window = MainWindow(args)
    window.setWindowIcon(QIcon('assets/vasco.png'))
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
