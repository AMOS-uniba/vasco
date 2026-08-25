#!/usr/bin/env python
"""
vasco as a filter: a fitting job on stdin, a reduction on stdout.

The window is for a person deciding which dots are stars. This is for a server that has already
decided: it has a plate that was good enough once, a list of dots the station's own software found,
and nobody to ask. It runs the same Matcher over the same catalogue and reports what it arrived at.

It holds no policy. Whether a residual is small enough to keep is the server's business -- it knows
which camera this is and what that camera is normally worth -- so this reports the residual and
says nothing about it. The only failures here are "there is nothing to fit" and "the optimiser
would not run".

Exit codes: 0 a result was produced (however poor), 1 the job could not be read or fitted, 2 the
job is not a job at all. stdout carries the result and nothing else; everything readable goes to
stderr.
"""
import argparse
import datetime
import logging
import math
import sys

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from demeteor.catalogue import Catalogue
from demeteor.projections import BorovickaProjection

from vasco import logger, yaml_io
from vasco.models.dotcollection import DotCollection
from vasco.models.matcher import Matcher
from vasco.models.sensordata import SensorData
from vasco.utilities import mask_sparse

JOB_FORMAT = 'amos-fit-job/1'
RESULT_FORMAT = 'amos-fit-result/1'
REDUCTION_FORMAT = 'amos-reduction/1'

#: What version stamps the result. The kernel-smoothed correction is a vector field that no twelve
#: numbers can hold, so the server cannot recompute it and records this instead: the same version,
#: given the same dots and the same baseline, lands on the same field.
try:
    from importlib.metadata import version

    SOFTWARE = f"vasco {version('vasco')}"
except Exception:                                       # not installed, running from a checkout
    SOFTWARE = "vasco (unreleased)"

#: The four constants that are angles, in the order BorovickaProjection takes its twelve. `a` is
#: not one -- it is the fraction by which the radius is stretched -- and neither are the radial
#: constants, which carry radians per millimetre or its powers.
ANGLES = ('a0', 'F', 'epsilon', 'E')

#: demeteor's constructor names, in order. The AMOS documents use lowercase for all twelve.
DEMETEOR = ('x0', 'y0', 'a0', 'A', 'F', 'V', 'S', 'D', 'P', 'Q', 'epsilon', 'E')
DOCUMENT = ('x0', 'y0', 'a0', 'a', 'f', 'v', 's', 'd', 'p', 'q', 'epsilon', 'e')

DEFAULTS = dict(method='raw', iterations=10000, pre_iterations=0, bandwidth=0.1,
                mask_low=10.0, mask_distant=0.5, sigma_clip=3.0, clip_rounds=2,
                min_stars=8)

log = logging.getLogger('vasco')


class JobError(ValueError):
    """ The job cannot be read, or names nothing to fit. """


def plate_from_degrees(constants: dict) -> BorovickaProjection:
    """
    A document's twelve into a projection. Degrees in, radians out, for the four that are angles.

    The single crossing between the two conventions on this side, mirroring
    Projection.from_degrees() on the server's. Doing one without the other is a silent error.
    """
    try:
        values = {name: float(constants[key]) for name, key in zip(DEMETEOR, DOCUMENT, strict=True)}
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError(f"the baseline plate cannot be read: {exc}") from exc

    for name in ANGLES:
        values[name] = math.radians(values[name])
    return BorovickaProjection(**values)


def plate_to_degrees(projection: BorovickaProjection) -> dict:
    """
    And back, for a document a person is going to read.

    Plain floats, not numpy scalars: as_tuple() hands back whatever the optimiser put in, and
    yaml.safe_dump refuses an np.float64 outright rather than writing the number.
    """
    values = dict(zip(DEMETEOR, projection.normalised().as_tuple(), strict=True))
    return {key: float(math.degrees(values[name]) if name in ANGLES else values[name])
            for name, key in zip(DEMETEOR, DOCUMENT, strict=True)}


def dots(entries: list, *, fnos: bool) -> DotCollection:
    """
    A job's dot list as vasco holds one. Millimetres, because the server scaled them.

    The server sends millimetres rather than pixels on purpose: Observation.get_scaling_shifter()
    is its authority on where the sensor's centre is and how big a pixel is, and the pixel size a
    station writes into its own file is a second opinion that has been seen to disagree. Handing
    the collection straight to SensorData and never calling set_shifter_scales leaves it alone.
    """
    if not entries:
        return DotCollection()

    xy = np.array([[float(dot['x']), float(dot['y'])] for dot in entries], dtype=float)
    intensity = np.array([float(dot.get('intensity') or 0) for dot in entries], dtype=float)
    numbers = (np.array([int(dot['fno']) for dot in entries], dtype=int) if fnos else None)
    return DotCollection(xy, intensity, fnos=numbers)


def build_matcher(job: dict) -> Matcher:
    location = job.get('location') or {}
    try:
        earth = EarthLocation(float(location['longitude']) * u.deg,
                              float(location['latitude']) * u.deg,
                              float(location['altitude']) * u.m)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError(f"the job names no usable location: {exc}") from exc

    try:
        when = datetime.datetime.fromisoformat(str(job['timestamp']))
    except (KeyError, ValueError) as exc:
        raise JobError(f"the job names no usable timestamp: {exc}") from exc
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)

    stars = dots(job.get('stars') or [], fnos=False)
    meteor = dots(job.get('meteor') or [], fnos=True)
    log.info(f"{stars.count} reference dots, {meteor.count} meteor frames, at {when}")

    sensor = SensorData(stars, meteor, location=earth, timestamp=when,
                        station=str(job.get('station') or 'unknown'))

    # The catalogue demeteor ships, never a file. There is nobody here to choose one, and a
    # setting nobody sets is a setting that goes stale.
    return Matcher(earth, Time(when), catalogue=Catalogue.bundled(), sensor_data=sensor)


def mask(matcher: Matcher, options: dict) -> None:
    """
    Throw away what a person would have thrown away by hand.

    Two limits, both in degrees. `mask_low` drops dots the plate puts near or below the horizon,
    where refraction is large and the roof is in the way. `mask_distant` drops catalogue stars that
    no dot came near, so that the pairing cannot reach for one halfway across the sky.
    """
    if (low := options['mask_low']) is not None:
        altitudes = matcher.sensor_data.stars.project(matcher.projection, masked=True,
                                                      flip_theta=True)[..., 0]
        matcher.mask_sensor_data(
            mask_sparse(matcher.sensor_data.stars, altitudes > math.radians(low)))
        log.info(f"above {low}: {matcher.sensor_data.stars.count_visible} of "
                 f"{matcher.sensor_data.stars.count} dots")

    if (distant := options['mask_distant']) is not None:
        nearest = np.min(matcher.distance_sky_all(masked=True), axis=0)
        matcher.mask_catalogue(
            mask_sparse(matcher.catalogue, nearest < math.radians(distant)))
        log.info(f"within {distant} of a dot: {matcher.catalogue.count_visible} of "
                 f"{matcher.catalogue.count} catalogue objects")


def residuals(matcher: Matcher) -> np.ndarray:
    """ Per-dot angular residual in degrees, for the dots still in the fit. """
    return np.degrees(matcher.position_errors_sky())


def optimise(matcher: Matcher, start: BorovickaProjection, options: dict) -> BorovickaProjection:
    """
    Fit, then throw out the dots that do not belong and fit again.

    The clipping has no counterpart in the window, where a person looks at the plot and masks the
    aeroplane. Without it one dot that is not a star pairs with whatever star is nearest and pulls
    the whole plate towards it -- and there is no one here to notice.
    """
    projection = start

    if options['pre_iterations']:
        projection = BorovickaProjection(
            *matcher.minimize(x0=np.array(projection.as_tuple()),
                              maxiter=options['pre_iterations']))
        matcher.update_pairing()

    for round_number in range(options['clip_rounds'] + 1):
        matcher.update_projection(projection)
        projection = BorovickaProjection(
            *matcher.minimize(x0=np.array(projection.as_tuple()), maxiter=options['iterations'],
                              callback=None))
        matcher.update_projection(projection)

        errors = residuals(matcher)
        log.info(f"round {round_number}: {errors.size} dots, "
                 f"rms {np.degrees(matcher.rms_error(np.radians(errors))):.5f} deg")

        if round_number == options['clip_rounds'] or errors.size == 0:
            break

        limit = options['sigma_clip'] * np.degrees(matcher.rms_error(np.radians(errors)))
        keep = errors <= limit
        if keep.all() or np.count_nonzero(keep) < options['min_stars']:
            log.info("nothing worth clipping" if keep.all()
                     else "clipping would leave too few dots, stopping")
            break

        log.info(f"clipping {np.count_nonzero(~keep)} dots beyond {limit:.5f} deg")
        matcher.mask_sensor_data(mask_sparse(matcher.sensor_data.stars, keep))
        matcher.update_pairing()

    return projection


def reduction(matcher: Matcher, projection: BorovickaProjection, job: dict,
              *, method: str, meteor: list) -> dict:
    """
    One amos-reduction/1 document, which is what the server already knows how to read.

    The kernel-smoothed document reports the *parametric* residual, the same number as the raw one,
    and that is deliberate rather than an oversight. The smoother is built from the very residuals
    the fit was left with, so it reproduces them almost exactly and a "residual after smoothing"
    would be near zero by construction -- a measure of how well an interpolator interpolates its
    own training data, not of how good the plate is. The number that means something about this
    camera on this night is the parametric one, and it is also the only one the server can check,
    since the twelve constants are all it stores.
    """
    errors = residuals(matcher)
    magnitudes = matcher.catalogue_vmag(masked=True)

    return {
        'format': REDUCTION_FORMAT,
        'identification': job.get('identification'),
        'software': SOFTWARE,
        'method': method,
        'baseline': job.get('baseline_code'),
        'projection': plate_to_degrees(projection),
        'fit': {
            'stars': int(errors.size),
            # Degrees, which is the unit Reduction.quality takes and what the window displays.
            # Matcher works in radians throughout; this is the one place it is converted.
            'residual_rms': float(np.degrees(matcher.rms_error(np.radians(errors))))
            if errors.size else None,
            'residual_max': float(np.max(errors)) if errors.size else None,
            'limiting_magnitude': float(np.max(magnitudes)) if magnitudes.size else None,
        },
        'meteor': meteor,
    }


def meteor_positions(matcher: Matcher, projection: BorovickaProjection,
                     *, corrected: bool) -> list:
    """
    Where the meteor was, frame by frame, keyed by the frame number the station gave it.

    By fno and not by position in the list, because DotCollection silently drops a dot whose
    intensity is not positive -- and Kvant reports a negative intensity often enough -- so the
    list is not necessarily the identification's frames in order. The server matches on fno.
    """
    if matcher.sensor_data.meteor.count == 0:
        return []

    matcher.update_projection(projection)
    positions = (matcher.correct_meteor_position(projection) if corrected
                 else matcher.project_meteor(projection))
    magnitudes = (matcher.correct_meteor_magnitude(projection, matcher._calibration) if corrected
                  else matcher._calibration(matcher.sensor_data.meteor.intensities(masked=False)))
    fnos = matcher.sensor_data.meteor.fnos(masked=False)

    return [
        {'fno': int(fno),
         'alt': float(alt), 'az': float(az % 360.0),
         'magnitude': float(magnitude) if np.isfinite(magnitude) else None}
        for fno, alt, az, magnitude
        in zip(fnos, positions.alt.degree, positions.az.degree, magnitudes, strict=True)
    ]


def fit(job: dict) -> dict:
    if str(job.get('format')) != JOB_FORMAT:
        raise JobError(f"expected a {JOB_FORMAT} job, got {job.get('format')!r}")

    options = {**DEFAULTS, **(job.get('options') or {})}
    baseline = plate_from_degrees(job.get('baseline') or {})

    matcher = build_matcher(job)
    if matcher.sensor_data.stars.count < options['min_stars']:
        raise JobError(f"{matcher.sensor_data.stars.count} reference dots is fewer than the "
                       f"{options['min_stars']} this job asks for; there is nothing to fit")

    matcher.update_projection(baseline)
    mask(matcher, options)
    matcher.update_pairing()

    projection = optimise(matcher, baseline, options)

    reductions = [reduction(matcher, projection, job, method='vasco',
                           meteor=meteor_positions(matcher, projection, corrected=False))]

    if options['method'] == 'kernel':
        # The parametric fit first and the field on top of it, which is why one process does both:
        # the smoother is built from the residuals the fit was left with.
        matcher.update_projection(projection)
        matcher.update_position_smoother(bandwidth=options['bandwidth'])
        matcher.update_magnitude_smoother(bandwidth=options['bandwidth'])
        reductions.append(reduction(matcher, projection, job, method='vasco-ks',
                                    meteor=meteor_positions(matcher, projection, corrected=True)))

    return {'format': RESULT_FORMAT, 'reductions': reductions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='vasco-fit',
        description="Fit an all-sky plate to reference dots. Reads an "
                    f"{JOB_FORMAT} document on stdin, writes an {RESULT_FORMAT} on stdout.")
    parser.add_argument('-d', '--debug', action='store_true', help="log every iteration")
    args = parser.parse_args(argv)

    log_ = logger.setupLog('vasco')
    log_.setLevel(logging.DEBUG if args.debug else logging.INFO)

    try:
        job = yaml_io.load(sys.stdin)
    except Exception as exc:
        print(f"vasco-fit: cannot read the job: {exc}", file=sys.stderr)
        return 2
    if not isinstance(job, dict):
        print("vasco-fit: the job is not a YAML mapping", file=sys.stderr)
        return 2

    try:
        result = fit(job)
    except JobError as exc:
        print(f"vasco-fit: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        log.error("the fit failed", exc_info=True)
        print(f"vasco-fit: the fit failed: {exc}", file=sys.stderr)
        return 1

    yaml_io.dump(result, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
