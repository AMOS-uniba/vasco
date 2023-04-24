import dotmap
from astropy.coordinates import EarthLocation
from astropy import units as u


class Station:
    def __init__(self, name, latitude, longitude, altitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def earth_location(self):
        return EarthLocation(self.longitude * u.deg, self.latitude * u.deg, self.altitude * u.m)


AMOS = dotmap.DotMap({
    'stations': dict(
        AGO =Station("AGO",                         48.37291,   17.27396,  531),
        ARBO=Station("Arborétum Tesárske Mlyňany",  48.32345,   18.36847,  201),
        KNM =Station("Kysucké Nové Mesto",          49.30734,   18.76538,  417),
        VAZ =Station("Važec",                       49.05443,   19.98989,  812),
        SC  =Station("Senec",                       48.22021,   17.39512,  138),
        SP  =Station("San Pedro de Atacama",       -22.95341,  -68.17934, 2403),
        PC  =Station("Paniri Caur",                -22.33535,  -68.64417, 2535),
        HK  =Station("Haleakala",                   20.70740, -156.25615, 3068),
        MK  =Station("Mauna Kea",                   19.82366, -155.47717, 4126),
        CE  =Station("Cederberg",                  -32.49940,   19.25276,  873),
        RC  =Station("Rogge Cloof",                -32.54822,   20.73456, 1550),
        FO  =Station("Forrest",                    -30.85805,  128.11504,  164),
        KY  =Station("Kybo",                       -31.02472,  126.59072,  176),
        MU  =Station("Mundrabilla",                -31.83560,  127.84869,   91),
        LP  =Station("La Palma",                    28.76002,  -17.88226, 2339),
        TE  =Station("Tenerife",                    28.30044,  -16.51224, 2416),
     )
})
