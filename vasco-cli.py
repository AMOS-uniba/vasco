#!/usr/bin/env python
import argparse
import argparsedirs
import logger
import logging
from pathlib import Path

from models import Catalogue
import colour as c

log = logger.setupLog('vasco')


class VascoCLI():
    def __init__(self):
        self.argparser = argparse.ArgumentParser(description="Virtual All-Sky CorrectOr, Command Line Interface",
                                                 formatter_class=argparse.RawTextHelpFormatter)
        self._add_arguments()
        self._process_arguments()

        self.load_catalogue(self.args.catalogue)
        self.process_files(self.args.infiles)

    def _add_arguments(self):
        self.argparser.add_argument('infiles', nargs='+', type=argparse.FileType('r'),
                                    help="input file(s)")
        self.argparser.add_argument('outdir', action=argparsedirs.WriteableDir,
                                    help="output directory")
        self.argparser.add_argument('-c', '--catalogue', required=True,
                                    help="star catalogue file (TSV)")
        self.argparser.add_argument('-p', '--parameters', type=argparse.FileType('r'),
                                    help="file with projection parameters (YAML)")
        self.argparser.add_argument('-u', '--pre-optimize', action='store_true', default=False,
                                    help="attempt to optimize the projection parameters before pairing")
        self.argparser.add_argument('--method', type=str, choices=['raw', 'kernel'], default='raw',
                                    help="selects a correction method\n"
                                        "    - raw: only projection parameters are optimized\n"
                                        "    - kernel: projection parameters are optimized and kernel smoothing method is applied")
        self.argparser.add_argument('-b', '--bandwidth', type=float, default=0.1,
                                    help="kernel smoother bandwidth (radians), default 0.1")
        self.argparser.add_argument('-i', '--iterations', type=int, default=250,
                                    help="number of optimization iterations, default 250")

        self.argparser.add_argument('-d', '--debug', action='store_true', default=False,
                                    help="enable debugging output")

    def _process_arguments(self):
        self.args = self.argparser.parse_args()
        self.outdir = Path(self.args.outdir)
        log.setLevel(logging.DEBUG if self.args.debug else logging.INFO)

    def load_catalogue(self, filename):
        self.catalogue = Catalogue.load(filename)

    def process_files(self, filenames):
        for file in filenames:
            log.info(f"Processing {c.path(file.name)}")

VascoCLI()