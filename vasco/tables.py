"""
A match, shaped for a table widget.

These were methods on Matcher, and they were the two that had no business being there: they return
a DotMap whose keys are the columns of a Qt table and whose units are degrees because that is what
a person reads. A library computing plate constants should not know that a column is called `px`.

Functions taking a matcher rather than methods on one, so the direction of the dependency says what
is going on: the window knows about the match, and the match knows nothing about the window.
"""
import logging

import dotmap
import numpy as np
from demeteor.projections import Projection

log = logging.getLogger('vasco')


def stars(matcher, projection: Projection) -> dotmap.DotMap:
    """
    One row per reference dot: where it is, where the plate puts it, and which star it was paired
    with.

    Unmasked throughout -- the table shows everything and greys out what is hidden, so the mask
    travels as a column rather than being applied.
    """
    log.debug("Building a stars model table")
    positions = matcher.sensor_data.stars.project(projection, masked=False)
    x = matcher.sensor_data.stars.xs(masked=False)
    y = matcher.sensor_data.stars.ys(masked=False)
    shifted = matcher.sensor_data.shifter.invert(x, y)

    return dotmap.DotMap(
        x=x,
        y=y,
        px=shifted[0],
        py=shifted[1],
        alt=np.degrees(positions[..., 0]),
        az=np.degrees(positions[..., 1]),
        intensity=matcher.sensor_data.stars.intensities(masked=False),
        star=matcher.pairing,
        # The pairing is an index into the catalogue, which is a number nobody can read
        name=matcher.catalogue.names(masked=False)[matcher.pairing],
        mask=matcher.sensor_data.stars.mask,
        count=matcher.sensor_data.stars.count,
        scalar_errors=np.degrees(matcher.distance_sky(masked=False)),
        vector_errors=np.degrees(matcher.vector_errors_full()),
        _dynamic=False,
    )


def catalogue(matcher) -> dotmap.DotMap:
    """ One row per catalogue object, where it is now and how bright. """
    log.debug("Building a catalogue model table")
    radec = matcher.catalogue.radec(matcher.location, matcher.time, masked=False)
    altaz = matcher.catalogue_altaz(masked=False)

    return dotmap.DotMap(
        catalogue=matcher.catalogue,
        name=matcher.catalogue.names(masked=False),
        dec=radec.dec.degree,
        ra=radec.ra.degree,
        alt=altaz.alt.degree,
        az=altaz.az.degree,
        vmag=matcher.catalogue.vmag(matcher.location, matcher.time, masked=False),
        mask=matcher.catalogue.mask,
        count=matcher.catalogue.count,
        _dynamic=False,
    )
