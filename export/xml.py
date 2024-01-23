from .base import Exporter


class XMLExporter(Exporter):
    """ XML based meteor exporter. Currently a semi-hardcoded mess but works for typical use cases. """

    def export(self, filename):
        with open(filename, 'w') as file:
            file.write(
f"""<?xml version="1.0" encoding="UTF-8" ?>
<ufoanalyzer_record version ="200"
    clip_name="{self._matcher.sensor_data.name}"
    o="1"
    y="{self._time.strftime("%Y")}"
    mo="{self._time.strftime("%m")}"
    d="{self._time.strftime("%d")}"
    h="{self._time.strftime("%H")}"
    m="{self._time.strftime("%M")}"
    s="{self._time.strftime('%S.%f')}"
    tz="0" tme="0" lid="{self._matcher.sensor_data.station}" sid="kvant"
    lng="{self._location.lon.value}" lat="{self._location.lat.value}" alt="{self._location.height.value}"
    cx="{self._matcher.sensor_data.rect.xmax}" cy="{self._matcher.sensor_data.rect.ymax}"
    fps="{self._matcher.sensor_data.fps}" interlaced="0" bbf="0"
    frames="{self._matcher.sensor_data.meteor.count}"
    head="{self._matcher.sensor_data.meteor.fnos(False)[0] - 1}"
    tail="0" drop="-1"
    dlev="0" dsize="0" sipos="0" sisize="0"
    trig="0" observer="{self._matcher.sensor_data.station}" cam="" lens=""
    cap="" u2="0" ua="0" memo=""
    az="0" ev="0" rot="0" vx="0"
    yx="0" dx="0" dy="0" k4="0"
    k3="0" k2="0" atc="0" BVF="0"
    maxLev="0" maxMag="0" minLev="0" mimMag="0"
    dl="0" leap="0" pixs="0" rstar="0"
    ddega="0" ddegm="0" errm="0" Lmrgn="0"
    Rmrgn="0" Dmrgn="0" Umrgn="0">
    <ua2_objects>
        <ua2_object
            fs="20" fe="64" fN="45" sN="45"
            sec="3" av="0" pix="0" bmax="0"
            bN="0" Lmax="0" mag="0" cdeg="0"
            cdegmax="0" io="0" raP="0" dcP="0"
            av1="0" x1="0" y1="0" x2="0"
            y2="0" az1="0" ev1="0" az2="0"
            ev2="0" azm="0" evm="0" ra1="0"
            dc1="0" ra2="0" dc2="0" ram="0"
            dcm="0" class="spo" m="0" dr="0"
            dv="0" Vo="0" lng1="0" lat1="0"
            h1="0" dist1="0" gd1="0" azL1="0"
            evL1="0" lng2="0" lat2="0" h2="0"
            dist2="0" gd2="0" len="0" GV="0"
            rao="0" dco="0" Voo="0" rat="0"
            dct="0" memo=""
            CodeRed="G"
            ACOM="324"
            sigma="0"
            sigma.azi="0"
            sigma.zen="0"
            A0="{self._projection.axis_shifter.a0}"
            X0="{self._projection.axis_shifter.x0}"
            Y0="{self._projection.axis_shifter.y0}"
            V="{self._projection.radial_transform.linear}"
            S="{self._projection.radial_transform.lin_coef}"
            D="{self._projection.radial_transform.lin_exp}"
            EPS="{self._projection.zenith_shifter.epsilon}"
            E="{self._projection.zenith_shifter.E}"
            A="{self._projection.axis_shifter.A}"
            F0="{self._projection.axis_shifter.F}"
            P="{self._projection.radial_transform.quad_coef}"
            Q="{self._projection.radial_transform.quad_exp}"
            C="1"
            CH1="0"
            CH2="0"
            CH3="0"
            CH4="0"
            magA="0"
            magB="0"
            magR2="0"
            magS="0"
            usingPrecession="False">
""")
            file.write(self.print_meteor())
            file.write("""
        </ua2_object>
    </ua2_objects>
</ufoanalyzer_record>""")

    def print_meteor(self):
        df = super()._get_meteor()
        return df.to_xml(index=False, root_name='ua2_objpath', row_name='ua2_fdata2',
                         xml_declaration=False, pretty_print=True,
                         attr_cols=['fno', 'b', 'bm', 'Lsum', 'mag', 'mag_r', 'az', 'ev', 'az_r', 'ev_r', 'ra', 'dec'])
