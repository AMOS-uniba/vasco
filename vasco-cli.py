#!/usr/bin/env python
import argparse
import argparsedirs
import logger
import logging
import numpy as np
import yaml
from pathlib import Path

from models import Catalogue, SensorData
from matchers import Matchmaker, Counselor
from projections import BorovickaProjection
import colour as c

log = logger.setupLog('vasco')


class VascoCLI:
    def __init__(self):
        self.catalogue = None

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
        self.argparser.add_argument('-p', '--parameters', type=argparse.FileType('r'), required=True,
                                    help="file with projection parameters (YAML)")
        self.argparser.add_argument('-u', '--pre-optimize', action='store_true', default=False,
                                    help="attempt to optimize the projection parameters before pairing")

        self.argparser.add_argument('-m', '--method', type=str, choices=['raw', 'kernel'], default='raw',
                                    help="selects a correction method\n"
                                    "    - raw: only projection parameters are optimized\n"
                                    "    - kernel: projection parameters are optimized and kernel smoothing is applied")
        self.argparser.add_argument('-b', '--bandwidth', type=float, default=0.1,
                                    help="kernel smoother bandwidth (radians), default 0.1")
        self.argparser.add_argument('-i', '--iterations', type=int, default=250,
                                    help="number of optimization iterations, default 250")
        self.argparser.add_argument('-k', '--mask-distant', nargs='?', type=float, const=0.5,
                                    help="mask catalogue stars that are more distant than this limit first")
        self.argparser.add_argument('-d', '--debug', action='store_true', default=False,
                                    help="enable debugging output")

    def _process_arguments(self):
        self.args = self.argparser.parse_args()
        print(self.args)

        self.outdir = Path(self.args.outdir)
        log.setLevel(logging.DEBUG if self.args.debug else logging.INFO)
        self.projection = BorovickaProjection.load(self.args.parameters)

        self.distance_limit = np.radians(self.args.mask_distant) if self.args.mask_distant is not None else 0

        log.info(f"Projection parameters loaded: {self.projection}")

    def load_catalogue(self, filename):
        self.catalogue = Catalogue.load(filename)

    def process_files(self, files):
        for file in files:
            self.process_file(file)

    def process_file(self, file: Path):
        log.info(f"Processing {c.path(file.name)}")
        sensor_data = SensorData.load_YAML(file.name)
        log.info(f"Sensor data loaded: {sensor_data}")

        matcher = Matchmaker(sensor_data.location, sensor_data.timestamp,
                             catalogue=self.catalogue, sensor_data=sensor_data)

        if self.args.pre_optimize:
            matcher.minimize(x0=np.array(self.projection.as_tuple()), maxiter=50)

        # Once the best match is found, pair the dots to the catalogue stars
        counselor = matcher.pair(self.projection)
        # ...and optimize the projection again (this should be very fast now)
        result = counselor.minimize(x0=np.array(self.projection.as_tuple()), maxiter=10000)
        projection = BorovickaProjection(*result)
        log.debug(projection)

        self.save_output_file((Path(self.args.outdir) / Path(file.name).stem).with_suffix('.yaml'), projection)

    def save_output_file(self, filename, projection):
        log.info(f"Saving output to {c.path(filename)}")
        with open(filename, 'w') as file:
            yaml.dump(dict(
                projection=dict(
                    name='Borovička',
                    parameters=projection.as_dict(),
                ),
            ), file)


vasco = VascoCLI()
