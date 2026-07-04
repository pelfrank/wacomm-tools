import sys
import numpy as np
from netCDF4 import Dataset
from util.WW33 import WW33
from util.Interpolator import Interp2D


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python " + str(sys.argv[0]) + " initialization_date source_file history_dir destination_file")
        sys.exit(-1)

    iDate = sys.argv[1]
    src = sys.argv[2]
    history_dir = sys.argv[3]
    dst = sys.argv[4]

    print("iDate:" + iDate + " src: " + src + " history: " + history_dir + " dst: " + dst)

    # Open the NetCDF file
    ncsrcfile = Dataset(src)

    # Read variables
    time = ncsrcfile.variables["time"][:]
    srcLats = ncsrcfile["latitude"][:]
    srcLons = ncsrcfile["longitude"][:]

    Xlon, Xlat = np.meshgrid(srcLons, srcLats)
    dLon = (srcLons[1]-srcLons[0])*.75
    dLat = (srcLats[1]-srcLats[0])*.75
    dstLat =  np.arange(Xlat.min(), Xlat.max(), dLat)
    dstLon =  np.arange(Xlon.min(), Xlon.max(), dLon)

    # Instantiate a WW33 archive file
    ww33 = WW33(dst, time, dstLon, dstLat)

    # Create a 2D biliniear interpolator on Rho points
    interpolator2D = Interp2D(Xlon, Xlat, dstLon, dstLat)

    print("dpt...")
    dpt = ncsrcfile.variables["dpt"][:]
    dpt = interpolator2D.interp(dpt)
    print("...dpt")

    print("hs...")
    hs = ncsrcfile.variables["hs"][:]
    hs = interpolator2D.interp(hs)
    print("...hs")

    print("lm...")
    lm = ncsrcfile.variables["lm"][:]
    lm = interpolator2D.interp(lm)
    print("...lm")

    print("fp...")
    fp = ncsrcfile.variables["fp"][:]
    fp = interpolator2D.interp(fp)
    print("...fp")

    print("dir...")
    dir = ncsrcfile.variables["dir"][:]
    dir = interpolator2D.interp(dir)
    print("...dir")

    print("t0m1...")
    t0m1 = ncsrcfile.variables["t0m1"][:]
    t0m1 = interpolator2D.interp(t0m1)
    print("...t0m1")

    print("Saving archive file...")
    ww33.dpt = dpt
    ww33.hs = hs
    ww33.lm = lm
    ww33.fp = fp
    ww33.dir = dir
    ww33.period = t0m1
    ww33.write()

    # Close the NetCDF file
    ncsrcfile.close()
    ww33.close()