



class SkyStarSet():
    """ A set of stars in alt-az format """


    def numpy(self):
        return np.stack()

    def altaz(self, location, time):
        altaz = AltAz(location=location, obstime=time, pressure=75000 * u.pascal, obswl=500 * u.nm)
        stars = self.stars.transform_to(altaz)
        return = np.stack((stars.alt.degree, stars.az.degree))


class Catalogue():
    def __init__():
        pass

    def load(self, filename):
        self.stars = pd.read_csv(filename, sep='\t', header=1)
        self.skycoords = SkyCoord(self.catalogue.ra * u.deg, self.catalogue.dec * u.deg)

    def filter(self, vmag)


    
