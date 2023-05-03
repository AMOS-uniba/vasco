import dotmap
from astropy.coordinates import EarthLocation
from astropy import units as u


class Station:
    def __init__(self, id, code, name, latitude, longitude, altitude):
        self.id = id
        self.code = code
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def earth_location(self):
        return EarthLocation(self.longitude * u.deg, self.latitude * u.deg, self.altitude * u.m)


AMOS = dotmap.DotMap({
    'stations': dict(
        AGO =Station( 1, "AGO", "AGO",                         48.37291,   17.27396,  531),
        ARBO=Station( 2, "ARBO","Arborétum Tesárske Mlyňany",  48.32345,   18.36847,  201),
        KNM =Station( 3, "KNM", "Kysucké Nové Mesto",          49.30734,   18.76538,  417),
        VAZ =Station( 4, "VAZ", "Važec",                       49.05443,   19.98989,  812),
        SC  =Station( 5, "SC",  "Senec",                       48.22021,   17.39512,  138),
        SP  =Station( 6, "SP",  "San Pedro de Atacama",       -22.95341,  -68.17934, 2403),
        PC  =Station( 7, "PC",  "Paniri Caur",                -22.33535,  -68.64417, 2535),
        HK  =Station( 8, "HK",  "Haleakala",                   20.70740, -156.25615, 3068),
        MK  =Station( 9, "MK",  "Mauna Kea",                   19.82366, -155.47717, 4126),
        CE  =Station(10, "CE",  "Cederberg",                  -32.49940,   19.25276,  873),
        RC  =Station(11, "RC",  "Rogge Cloof",                -32.54822,   20.73456, 1550),
        FO  =Station(12, "FO",  "Forrest",                    -30.85805,  128.11504,  164),
        KY  =Station(13, "KY",  "Kybo",                       -31.02472,  126.59072,  176),
        MU  =Station(14, "MU",  "Mundrabilla",                -31.83560,  127.84869,   91),
        LP  =Station(15, "LP",  "La Palma",                    28.76002,  -17.88226, 2339),
        TE  =Station(16, "TE",  "Tenerife",                    28.30044,  -16.51224, 2416),
     )
})
