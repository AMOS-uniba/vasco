import logging
import time

import colour as c


class VascoFormatter(logging.Formatter):
    def __init__(self):
        super().__init__('{asctime} {levelname} {message}', "%H:%M:%S", '{')

    def format(self, record):
        record.levelname = {
            'DEBUG':    c.debug,
            'INFO':     c.ok,
            'WARNING':  c.warn,
            'ERROR':    c.err,
            'CRITICAL': c.critical,
        }[record.levelname](record.levelname)
        return super().format(record)

    def formatTime(self, record, format):
        ct = self.converter(record.created)
        return f"{time.strftime('%H:%M:%S', ct)}.{int(record.msecs):03d}"


def setupLog(name, **kwargs):
    formatter = VascoFormatter()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False

    return log
