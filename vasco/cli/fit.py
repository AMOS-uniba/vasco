#!/usr/bin/env python
"""
vasco as a filter: a fitting job on stdin, a reduction on stdout.

The window is for a person deciding which dots are stars. This is for a caller that has already
decided, and it is almost nothing now: the fit itself is `demeteor.fitting`, where both this and the
AMOS server can reach it. What is left here is a command line and two YAML documents.

It holds no policy about quality. Whether a residual is small enough to keep is the caller's
business -- the caller knows which camera this is and what that camera is normally worth.

Exit codes: 0 a result was produced (however poor), 1 the job could not be read or fitted, 2 the
job is not a job at all. stdout carries the result and nothing else; everything readable goes to
stderr.
"""
import argparse
import logging
import sys

from demeteor.fitting import JOB_FORMAT, RESULT_FORMAT, JobError, fit

from vasco import logger, yaml_io

log = logging.getLogger('vasco')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='vasco-fit',
        description="Fit an all-sky plate to reference dots. Reads an "
                    f"{JOB_FORMAT} document on stdin, writes an {RESULT_FORMAT} on stdout.")
    parser.add_argument('-d', '--debug', action='store_true', help="log every iteration")
    args = parser.parse_args(argv)

    log_ = logger.setupLog('vasco')
    log_.setLevel(logging.DEBUG if args.debug else logging.INFO)
    # The fit logs as demeteor, being demeteor's now, and its messages are the interesting ones:
    # how many dots survived masking, what was clipped, which bandwidth was chosen.
    demeteor_log = logging.getLogger('demeteor')
    demeteor_log.handlers = log_.handlers
    demeteor_log.setLevel(log_.level)

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
