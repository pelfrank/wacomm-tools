import sys
import numpy as np
from netCDF4 import Dataset
from util.Interpolator import Interp2D, depths
from util.Distributor import Distrib3D
from util.Wacomm import Wacomm


def compute_sfconc(conc, depth_limit, mask2d, depths, fill_value=1e37):
    """
    Compute the vertical sum of concentration up to a given depth,
    ignoring fill values and masked areas.

    Parameters:
        conc (np.ndarray): 3D concentration array with shape (depth, lat, lon).
        depth_limit (float): Maximum depth (in meters) for the integration.
        mask2d (np.ndarray): 2D mask of shape (lat, lon) with 1 for water, 0 for land.
        depths (list or np.ndarray): List of depth levels corresponding to the first axis of `conc`.
        fill_value (float): Value used to indicate missing or invalid data.

    Returns:
        np.ndarray: 2D array (lat, lon) containing the summed concentration
                    from the surface down to `depth_limit`.
    """
    # Ensure depths is a NumPy array
    depth_arr = np.array(depths)

    # Create boolean mask of levels <= depth_limit
    depth_mask = depth_arr <= depth_limit  # shape (n_depths,)

    # Expand spatial mask to 3D and combine with depth mask
    mask3d = depth_mask[:, None, None] * mask2d[None, :, :]

    # Replace fill values with zero so they don’t inflate the sum
    conc_clean = np.where(conc == fill_value, 0.0, conc)

    # Sum along the depth axis
    sfconc = np.sum(conc_clean * mask3d, axis=0)  # shape (lat, lon)

    # Restore fill_value on land points
    sfconc[mask2d == 0] = fill_value

    return sfconc


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
    s_rho = ncsrcfile["s_rho"][:]
    mask_rho = ncsrcfile["mask_rho"][:]
    H = ncsrcfile["h"][:]

    dstLon = np.linspace(Xlon.min(), Xlon.max(), len(Xlon[0]))
    dstLat = np.linspace(Xlat.min(), Xlat.max(), len(Xlat))

    # Instantiate a Wacomm archive file
    wacomm = Wacomm(dst, time, depths, dstLon, dstLat)

    # Create a 2D biliniear interpolator on Rho points
    interpolator2DRho = Interp2D(Xlon, Xlat, dstLon, dstLat)

    # Create a 3D distributor on Rho points
    distributor3DRho = Distrib3D(Xlon, Xlat, dstLon, dstLat, s_rho, mask_rho, H)

    print("conc...")
    conc = ncsrcfile.variables["conc"][:]
    conc = distributor3DRho.distrib(conc)
    print("...conc")

    print("sfconc...")
    sfconc = conc[0, 0]
    sfconc_10m = compute_sfconc(conc[0], 10.0, distributor3DRho.mask, depths)
    sfconc_30m = compute_sfconc(conc[0], 30.0, distributor3DRho.mask, depths)
    print("...sfconc")

    print("Saving archive file...")
    wacomm.mask = distributor3DRho.mask
    wacomm.conc = conc
    wacomm.sfconc = sfconc
    wacomm.sfconc_10m = sfconc_10m
    wacomm.sfconc_30m = sfconc_30m
    wacomm.write()

    # Close the NetCDF file
    ncsrcfile.close()
    wacomm.close()

