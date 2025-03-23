import logging
import math
import dotmap
import numpy as np
import scipy as sp

from typing import Callable, Optional
from pathlib import Path

from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time

from amosutils.catalogue import Catalogue
from amosutils.projections import Projection, BorovickaProjection
from amosutils.metrics import spherical, euclidean
from numpy.typing import NDArray

from correctors import KernelSmoother, kernels
from photometry import Calibration
from models.sensordata import SensorData
from utilities import hash_numpy, proj_to_disk, altaz_to_disk, disk_to_altaz, unit_grid, numpy_to_disk

log = logging.getLogger('vasco')


class Matcher:
    """
    The base class for matching sensor data to the catalogue.
    """

    def __init__(self,
                 location: EarthLocation,
                 time: Time,
                 projection_cls: type[Projection] = BorovickaProjection,
                 *,
                 catalogue: Optional[Catalogue] = None,
                 sensor_data: Optional[SensorData] = None):
        self._altaz: Optional[AltAz] = None
        self._altaz_numpy: Optional[np.ndarray] = None

        self._cached_distances = None
        self._hash_observed = None
        self._hash_catalogue = None

        self.pairing: Optional[np.ndarray] = None
        self.pairing_fixed: bool = False

        self.position_smoother = KernelSmoother(np.zeros(shape=(1, 2), dtype=float),
                                                np.zeros(shape=(1, 2), dtype=float))
        self.magnitude_smoother = KernelSmoother(np.zeros(shape=(1, 2), dtype=float),
                                                 np.ndarray(shape=(1,), dtype=float))

        self.projection_cls: type[Projection] = projection_cls
        self._projection: Optional[Projection] = None

        self.location: Optional[EarthLocation] = None
        self.time: Optional[Time] = None
        self.catalogue = Catalogue() if catalogue is None else catalogue
        self.sensor_data = SensorData() if sensor_data is None else sensor_data
        self.update_location_time(location, time)

        log.debug(f"Created a Matcher ({self.projection_cls.__qualname__})")

    def load_catalogue(self, filename: Path):
        catalogue = Catalogue(filename)
        catalogue.build_planets(self.location, self.time)
        # ToDo: Some form of catalogue validation should come here
        self.catalogue = catalogue
        self.invalidate_altaz()
        log.info(f"Loaded a catalogue from {filename}: {self.catalogue.count} stars")

    @property
    def valid(self) -> bool:
        return self.catalogue is not None and self.sensor_data is not None

    def mask_catalogue(self, mask):
        self.catalogue.mask &= mask
        self.invalidate_altaz()
        self.update_pairing()

    def mask_sensor_data(self, mask):
        self.sensor_data.stars.mask &= mask
        self.update_pairing()

    def update_location_time(self, location, time):
        """
        Update internal location and time and request the catalogue to recompute the stars' positions.
        """
        self.location = location
        self.time = time

        if self.catalogue.populated:
            self.invalidate_altaz()

    def update_projection(self, projection: Projection):
        self._projection = projection
        if not self.pairing_fixed:
            self.update_pairing()

    def invalidate_altaz(self):
        log.debug("Invalidating the cached altaz")
        self._altaz = None

    def catalogue_altaz(self, *, masked: bool) -> AltAz:
        if self._altaz is None:
            log.debug(f"Requesting catalogue for {self.location}, {self.time}, {masked}")
            self._altaz = self.catalogue.altaz(self.location, self.time, masked=False)
        else:
            log.debug(f"Requesting catalogue for {self.location}, {self.time}, {masked} (using cached)")

        if masked:
            return self._altaz[self.catalogue.mask]
        else:
            return self._altaz

    def catalogue_altaz_np(self, *, masked: bool) -> np.ndarray:
        """
        Return the current masked catalogue altaz as a numpy array
        """

        altaz = self.catalogue_altaz(masked=masked)
        return np.array([altaz.alt.radian, altaz.az.radian], dtype=float).T

    def catalogue_altaz_paired(self) -> np.ndarray:
        return self.catalogue_altaz_np(masked=False)[self.pairing]

    def catalogue_vmag(self, *, masked: bool):
        """
        Return the current catalogue vmag as a numpy array
        """
        log.debug("Requesting a new magnitude catalogue")
        vmag = self.catalogue.vmag(self.location, self.time, masked=masked)
        return vmag

    def catalogue_vmag_paired(self) -> np.ndarray:
        return self.catalogue_vmag(masked=False)[self.pairing]

    def compute_pairing(self) -> np.ndarray:
        """
        Project the sensor data onto the sky and find the nearest star for every dot.

        Returns
            array(M): index of the star nearest to each dot, or NaN if not applicable
        """
        obs = self.sensor_data.stars.project(self._projection, masked=False, flip_theta=True)
        cat = self.catalogue_altaz_np(masked=True)

        # Now add an extra axis to compute the Cartesian product
        obs = np.expand_dims(obs, 1)
        cat = np.expand_dims(cat, 0)
        dist = spherical(obs, cat)

        if dist.size > 0:
            pairing = np.argmin(dist, axis=1)
        else:
            empty = np.empty(shape=(dist.shape[0]), dtype=int)
            empty[...] = np.nan
            pairing = empty

        return pairing

    def update_pairing(self) -> None:
        """
        Compute the current pairing and use it from now on
        """
        log.debug("Updating the pairing")
        idx = np.arange(0, self.catalogue.count)[self.catalogue.mask]
        self.pairing = idx[self.compute_pairing()]

    def fix_pairing(self, fix: bool) -> None:
        self.pairing_fixed = fix

    def _compute_distances_sky(self,
                               observed: np.ndarray[float],
                               catalogue: np.ndarray[float],
                               *,
                               paired: bool) -> np.ndarray:
        """
        Compute distances in the sky of `observed` and `catalogue` objects
        from the full Cartesian product

        Parameters
        ----------
        observed:   np.ndarray(M, 2)
        catalogue:  np.ndarray(N, 2)

        Returns
        -------
        np.ndarray(M, N)
        """

        return spherical(observed, catalogue[self.pairing[self.sensor_data.stars.mask]])
        #else:
        #    # Binary and is here to *prevent* short circuit, otherwise it can fail!
        #    if (((hash_observed := hash_numpy(observed)) == self._hash_observed) &
        #        ((hash_catalogue := hash_numpy(catalogue)) == self._hash_catalogue)):
        #        log.debug(f"Returning cached")
        #    else:
        #        self._hash_observed = hash_observed
        #        self._hash_catalogue = hash_catalogue

        #        observed = np.expand_dims(observed, 1)
        #        catalogue = np.expand_dims(catalogue, 0)
        #        self._cached_distances = spherical(observed, catalogue)
        #        log.debug(f"Computing sky distance for {observed.shape} × {catalogue.shape}")
        #    return self._cached_distances

    def distance_sky(self, *, masked: bool = True) -> np.ndarray:
        obs = self.sensor_data.stars.project(self._projection, masked=masked, flip_theta=True)
        cat = self.catalogue_altaz_paired()
        if masked:
            cat = cat[self.sensor_data.stars.mask]
        log.debug(f"Distance in the sky: {obs.shape}, {cat.shape}")
        return spherical(obs, cat)

    def distance_sky_full(self) -> np.ndarray:
        obs = self.sensor_data.stars.project(self._projection, masked=False, flip_theta=True)
        cat = self.catalogue_altaz_np(masked=False)
        return spherical(np.expand_dims(obs, 1), np.expand_dims(cat, 0))

    def vector_errors(self) -> np.ndarray:
        obs = self.sensor_data.stars.project(self._projection, masked=True, flip_theta=True)
        cat = self.catalogue_altaz_paired()[self.sensor_data.stars.mask]
        log.debug(f"Vector difference in the sky: {obs.shape}, {cat.shape}")
        return obs - cat

    def vector_errors_full(self) -> np.ndarray:
        obs = self.sensor_data.stars.project(self._projection, masked=False, flip_theta=True)
        cat = self.catalogue_altaz_paired()
        log.debug(f"Vector difference in the sky: {obs.shape}, {cat.shape}")
        return obs - cat

    def position_errors_sky(self) -> np.ndarray:
        return self.distance_sky()

    def magnitude_errors_sky(self,
                             calibration: Calibration,  # ToDo: also move calibration out
                             ) -> np.ndarray:
        obs = calibration(self.sensor_data.stars.intensities(masked=True))
        cat = self.catalogue_vmag_paired()[self.sensor_data.stars.mask]
        return obs - cat

    def update_position_smoother(self, *, bandwidth: float = 0.1):
        log.debug(f"Updating position smoother (bandwidth = {bandwidth}")
        cat = numpy_to_disk(self.catalogue_altaz_paired())[self.sensor_data.stars.mask]
        obs = proj_to_disk(self.sensor_data.stars.project(self._projection, masked=True))
        self.position_smoother = KernelSmoother(
            obs, obs - cat,
            kernel=kernels.nexp,
            bandwidth=bandwidth,
        )

    def update_magnitude_smoother(self, calibration: Calibration, *, bandwidth: float = 0.1):
        log.debug(f"Updating magnitude smoother (bandwidth = {bandwidth}")
        mcat = self.catalogue_vmag_paired()[self.sensor_data.stars.mask]
        obs = proj_to_disk(self.sensor_data.stars.project(self._projection, masked=True))
        mobs = calibration(self.sensor_data.stars.intensities(masked=True))
        self.magnitude_smoother = KernelSmoother(
            obs, np.expand_dims(mobs - mcat, 1),
            kernel=kernels.nexp,
            bandwidth=bandwidth,
        )

    @staticmethod
    def rms_error(errors: np.ndarray) -> float:
        if errors.size == 0:
            return np.nan
        else:
            return np.sqrt(np.sum(np.square(errors)) / errors.size)

    @staticmethod
    def max_error(errors: np.ndarray) -> float:
        return np.max(errors, initial=0)

    @staticmethod
    def find_nearest_value(dist, *, axis: int) -> np.ndarray:
        """
        Find the nearest dot to star or vice versa

        axis: int
            0 for the nearest star to every dot
            1 for the nearest dot to every star
        """
        return np.min(dist, axis=axis, initial=np.pi)

    @staticmethod
    def find_nearest_index(dist, *, axis: int) -> np.ndarray:
        """
        Find the index of the nearest dot to star or vice versa

        axis: int
            0 for the nearest star to every dot
            1 for the nearest dot to every star
        """
        if dist.size > 0:
            return np.argmin(dist, axis=axis)
        else:
            return np.empty(shape=(dist.shape[axis], 0), dtype=int)

    def correct_meteor(self, projection: Projection, calibration: Calibration) -> dotmap.DotMap:
        log.debug("Computing vector correction for the meteor")
        positions_raw = self.project_meteor(projection)
        positions_corrected = self.correct_meteor_position(projection)
        positions_correction_angle = positions_raw.separation(positions_corrected)
        positions_correction_xy = self.correction_meteor_xy(projection)

        intensities_raw = self.sensor_data.meteor.intensities(masked=False)
        intensities_corrected = self.correct_meteor_magnitude(projection, calibration)
        magnitudes_raw = calibration(intensities_raw)
        magnitudes_corrected = intensities_corrected
        magnitudes_correction = self.correction_meteor_mag(projection)

        return dotmap.DotMap(
            position_raw=positions_raw,
            position_corrected=positions_corrected,
            magnitudes_raw=magnitudes_raw,
            magnitudes_corrected=magnitudes_corrected,
            positions_correction_xy=positions_correction_xy,
            positions_correction_angle=positions_correction_angle,
            magnitudes_correction=magnitudes_correction,
            count=self.sensor_data.meteor.count,
            fnos=self.sensor_data.meteor.fnos(masked=False),
            _dynamic=False,
        )

    def _meteor_xy(self, projection):
        """ Return on-disk xy coordinates for the meteor after applying the projection """
        return proj_to_disk(self.sensor_data.meteor.project(projection, masked=False, flip_theta=True))

    def project_meteor(self, projection: Projection):
        return disk_to_altaz(self._meteor_xy(projection))

    def correction_meteor_xy(self, projection: Projection):
        return self.position_smoother(self._meteor_xy(projection))

    def correction_meteor_mag(self, projection: Projection) -> NDArray:
        return np.ravel(self.magnitude_smoother(self._meteor_xy(projection)))

    def correct_meteor_position(self, projection: Projection) -> AltAz:
        return disk_to_altaz(self._meteor_xy(projection) - self.correction_meteor_xy(projection))

    def correct_meteor_magnitude(self, projection: Projection, calibration: Calibration) -> np.ndarray:
        return calibration(self.sensor_data.meteor.intensities(masked=False)) - self.correction_meteor_mag(projection)

    @staticmethod
    def _grid(smoother, resolution=21, *, masked: bool):
        xx, yy = unit_grid(resolution, masked=masked)
        nodes = np.ma.stack((xx.ravel(), yy.ravel()), axis=1)
        return smoother(nodes).reshape(resolution, resolution, -1)

    def position_grid(self, resolution=21):
        return self._grid(self.position_smoother, resolution, masked=True)

    def magnitude_grid(self, resolution=21):
        return self._grid(self.magnitude_smoother, resolution, masked=False)

    def _build_optimization_function(self,
                                     mask: np.ndarray[float]) -> Callable[[np.ndarray[float], ...], float]:
        """
        Split the parameter vector into immutable and variable part depending on mask.
        Return a loss function in which only variable parameters are to be optimized
        and immutable ones are treated as constants
        """
        ifixed = np.where(~mask)
        ivariable = np.where(mask)

        def func(x: np.ndarray[float], *args) -> float:
            variable = np.array(x)

            vec = np.zeros(shape=(12,))
            np.put(vec, ivariable, variable)
            np.put(vec, ifixed, np.array(args))

            self._projection = self.projection_cls(*vec)
            return self.rms_error(self.position_errors_sky(axis=1, mask_catalogue=True, mask_sensor=True))

        return func

    def get_optimization_bounds(self, mask):
        return self.projection_cls.bounds[mask]

    def minimize(self,
                 x0=np.array((0, 0, 0, 0, math.tau / 4, 1, 0, 0, 0, 0, 0, 0)),
                 maxiter=30,
                 *,
                 mask=np.ones(shape=(12,), dtype=bool),
                 callback: Callable = lambda x: log.debug(x)):
        func = self._build_optimization_function(mask)
        args = np.array(x0)[~mask]

        if np.count_nonzero(mask) == 0:
            log.warning("At least one parameter must be allowed to vary")
            return tuple(x0)

        # Do the actual optimization over the selected values
        result = sp.optimize.minimize(
            func,
            x0[mask],
            args,
            method='Nelder-Mead',
            bounds=self.get_optimization_bounds(mask),
            options=dict(maxiter=maxiter, disp=True),
            callback=callback,
        )

        # Restore the full parameter vector from immutable original args and variable optimized args
        vec = np.zeros(shape=(12,))
        np.put(vec, np.where(mask), result.x)
        np.put(vec, np.where(~mask), args)

        return tuple(vec)
