# Star catalogue bundled with demeteor

## `HYG44.tsv`

A subset of the **HYG star database, version 4.4**, by **astronexus**.

- Source: <https://codeberg.org/astronexus/hyg>
  (previously <https://github.com/astronexus/HYG-Database>, which is no longer updated)
- Licence: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**,
  <https://creativecommons.org/licenses/by-sa/4.0/>

HYG combines the HIPPARCOS, Yale Bright Star and Gliese catalogues into one dataset. Versions from
4.0 onwards are licensed CC BY-SA 4.0; earlier versions were CC BY-SA 2.5.

### Changes made to it

This file is a modified version of the original, not a copy of it:

- reduced to the 5070 stars of visual magnitude 6.00 or brighter, which is roughly what an all-sky
  camera records, with the Sun dropped;
- reduced to six of the original columns — `ra`, `dec`, `dist`, `vmag`, `absmag`, `name`;
- rewritten as tab-separated values with a comment line in front of the header;
- ordered brightest first;
- stars with no proper name in HYG carry the literal `unnamed` in the `name` column, which is 4642
  of the 5070. `dist` is HYG's sentinel of 100000 parsecs on the 104 stars with no parallax, and
  `absmag` on those rows is derived from that sentinel rather than measured.

`tools/build_catalogue.py` in this repository does exactly these steps, so the file can be rebuilt
from a later HYG release without rediscovering the conventions.

### What ShareAlike means here

CC BY-SA 4.0 is a copyleft licence, and it attaches to **this data file and anything adapted from
it**, which must stay under CC BY-SA 4.0 or a compatible licence. It says nothing about the licence
of demeteor's own code, which merely reads the file.

If you redistribute this file, or a catalogue derived from it, carry this notice with it.

### A note on the version

This started as 4.2. Within the magnitude-limited subset, going to 4.4 changed 60 names, four
visual and five absolute magnitudes, one distance, and one right ascension by 0.9 mas. The names
are a net gain of 44 -- 428 stars now have one against 384 -- almost all from the IAU list; three
went the other way, of which Capella's is a correction, the name moving from the companion to the
primary. It also fixed `Yunü`, which 4.2 carried as `YunÃ¼`: UTF-8 read as Latin-1 somewhere
upstream.
