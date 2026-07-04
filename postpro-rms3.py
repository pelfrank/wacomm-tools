import sys
import numpy as np
from netCDF4 import Dataset
from util.Interpolator import Interp2D, Interp3D, depths
from util.ROMS import ROMS


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
    time = ncsrcfile.variables["ocean_time"][:]
    Xlat = ncsrcfile["lat_rho"][:]
    Xlon = ncsrcfile["lon_rho"][:]
    Ulat = ncsrcfile["lat_u"][:]
    Ulon = ncsrcfile["lon_u"][:]
    Vlat = ncsrcfile["lat_v"][:]
    Vlon = ncsrcfile["lon_v"][:]
    s_rho = ncsrcfile["s_rho"][:]
    mask_rho = ncsrcfile["mask_rho"][:]
    mask_u = ncsrcfile["mask_u"][:]
    mask_v = ncsrcfile["mask_v"][:]
    H = ncsrcfile["h"][:]

    dstLon = np.linspace(Xlon.min(), Xlon.max(), len(Xlon[0]))
    dstLat = np.linspace(Xlat.min(), Xlat.max(), len(Xlat))

    # Instantiate a ROMS archive file
    roms = ROMS(dst, time, depths, dstLon, dstLat)

    # Create a 2D biliniear interpolator on Rho points
    interpolator2DRho = Interp2D(Xlon, Xlat, dstLon, dstLat)
    # Create a 2D biliniear interpolator on U points
    interpolator2DU = Interp2D(Ulon, Ulat, dstLon, dstLat)
    # Create a 2D biliniear interpolator on V points
    interpolator2DV = Interp2D(Vlon, Vlat, dstLon, dstLat)

    # Create a 3D biliniear interpolator on Rho points
    interpolator3DRho = Interp3D(Xlon, Xlat, dstLon, dstLat, s_rho, mask_rho, H)
    # Create a 3D biliniear interpolator on U points
    interpolator3DU = Interp3D(Ulon, Ulat, dstLon, dstLat, s_rho, mask_u, H)
    # Create a 3D biliniear interpolator on V points
    interpolator3DV = Interp3D(Vlon, Vlat, dstLon, dstLat, s_rho, mask_v, H)

    print("h...")
    H = interpolator2DRho.interp(H)
    print("...h")

    print("zeta...")
    zeta = ncsrcfile["zeta"][:]
    zeta = interpolator2DRho.interp(zeta)
    print("...zeta")

    print("temp...")
    temp = ncsrcfile["temp"][:]
    temp = interpolator3DRho.interp(temp)
    print("...temp")

    print("temp at bottom...")
    temp_at_bottom = interpolator3DRho.bottomValues(temp)
    print("...temp at bottom")

    print("temp at surface...")
    temp_at_surface = interpolator3DRho.surfaceValues(temp)
    print("...temp at surface")

    print("salt...")
    salt = ncsrcfile["salt"][:]
    salt = interpolator3DRho.interp(salt)
    print("...salt")

    print("salt at bottom...")
    salt_at_bottom = interpolator3DRho.bottomValues(salt)
    print("...salt at bottom")

    print("salt at surface...")
    salt_at_surface = interpolator3DRho.surfaceValues(salt)
    print("...salt at surface")

    print("U...")
    u = ncsrcfile["u"][:]
    u = interpolator3DU.interp(u)
    print("...U")

    print("U at bottom...")
    U_at_bottom = interpolator3DRho.bottomValues(u)
    print("...U at bottom")

    print("U at surface...")
    U_at_surface = interpolator3DRho.surfaceValues(u, factor=1.2)
    print("...U at surface")

    print("ubar...")
    ubar = ncsrcfile["ubar"][:]
    ubar = interpolator2DU.interp(ubar)
    print("...ubar")

    print("V...")
    v = ncsrcfile["v"][:]
    v = interpolator3DV.interp(v)
    print("...V")

    print("V at bottom...")
    V_at_bottom = interpolator3DRho.bottomValues(v)
    print("...V at bottom")

    print("V at surface...")
    V_at_surface = interpolator3DRho.surfaceValues(v, factor=1.2)
    print("...V at surface")

    print("vbar...")
    vbar = ncsrcfile["vbar"][:]
    vbar = interpolator2DV.interp(vbar)
    print("...vbar")

    print("Saving archive file...")
    roms.h = H
    roms.temp = temp
    roms.tempBottom = temp_at_bottom
    roms.tempSurface = temp_at_surface
    roms.salt = salt
    roms.saltBottom = salt_at_bottom
    roms.saltSurface = salt_at_surface
    roms.zeta = zeta
    roms.U = u
    roms.uBottom = U_at_bottom
    roms.uSurface = U_at_surface
    roms.V = v
    roms.vBottom = V_at_bottom
    roms.vSurface = V_at_surface
    roms.ubar = ubar
    roms.vbar = vbar
    roms.write()

    # Close the NetCDF file
    ncsrcfile.close()
    roms.close()
