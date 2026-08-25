"""
How vasco reads and writes YAML: version 1.2, never 1.1.

PyYAML implements YAML 1.1, in which a leading zero means octal. Every AMOS sighting file
zero-pads its frame numbers -- `fno: 015` -- and 27 of the files in data/ do. Read as 1.1 that
frame is number 13 and `fno: 020` is 16, while `fno: 018` is not valid octal at all and comes back
as the string '018', which int() turns into 18. So a trail of frames 15 to 34 arrives as
13, 14, 15, 18, 19, 16, ... -- 18 and 19 twice each, and six frame numbers that never existed.

Nothing downstream can notice. The dots are in the right places and only their labels are wrong,
so the fit is unaffected and the plots look right -- but the frame numbers are what tie a sky
position back to the frame it came from, and they are what the server matches a reduction's
positions on. A duplicate silently overwrote one.

ruamel in safe mode is YAML 1.2, where a leading zero is a leading zero. The server settled this
the same way, for the same reason, in Metadata._load_yaml.
"""
import io
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml import YAMLError            # noqa: F401  (re-exported: callers catch this)


def _reader() -> YAML:
    return YAML(typ='safe')


def _writer() -> YAML:
    writer = YAML(typ='safe')
    writer.default_flow_style = False
    return writer


def load(stream) -> Any:
    """ Parse a stream, a path or a string. """
    if isinstance(stream, (str, bytes)) and not isinstance(stream, bytes):
        # A path, not a document: every caller here passes a filename or an open file
        with open(stream) as file:
            return _reader().load(file)
    return _reader().load(stream)


def dump(data: Any, stream) -> None:
    _writer().dump(data, stream)


def dumps(data: Any) -> str:
    out = io.StringIO()
    dump(data, out)
    return out.getvalue()
