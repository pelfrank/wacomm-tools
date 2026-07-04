import numpy as np
from numba import jit
from scipy.interpolate import griddata


depths = [3.1657474, 5.4649634, 7.9203773, 10.536604, 13.318384, 16.270586, 19.39821, 22.706392, 26.2004, 29.885643,
          33.767673, 37.852192, 42.14504, 46.65221, 51.37986, 56.334286, 61.521957, 66.94949, 72.62369, 78.5515,
          84.74004, 91.19663, 97.92873, 104.94398, 112.250206, 119.85543, 127.76784, 135.9958, 144.5479, 153.43285,
          162.65962, 172.23735, 182.17535, 192.48314, 203.17044, 214.24716, 225.7234, 237.60947, 249.91585, 262.65323,
          275.83252, 289.46478, 303.5613, 318.13354, 333.19315, 348.75195, 364.82196, 381.41544, 398.5447, 416.22232,
          434.46106, 453.27377, 472.6735, 492.67346, 513.287, 534.5276, 556.4089, 578.9446, 602.1486, 626.0349,
          650.61755, 675.9107, 701.92865, 728.6856, 756.19604, 784.4743, 813.53485, 843.39215, 874.06067, 905.5548,
          937.8891, 971.0779, 1005.1355, 1040.0763, 1075.9143, 1112.6637, 1150.3384, 1188.9521, 1228.5188, 1269.0518,
          1310.5642, 1353.0693, 1396.58, 1441.1086, 1486.6678, 1533.2694, 1580.9252, 1629.6466, 1679.4448, 1730.3303,
          1782.3136, 1835.4045, 1889.6127, 1944.9471, 2001.4166, 2059.029, 2117.7925, 2177.714, 2238.8003, 2301.0576,
          2364.4917, 2429.1077, 2494.9102, 2561.903, 2630.0898, 2699.4736, 2770.0566, 2841.8408, 2914.827, 2989.0159,
          3064.4075, 3141.0015, 3218.7961, 3297.7903, 3377.9814, 3459.3662, 3541.942, 3625.7039, 3710.6475, 3796.768,
          3884.0596, 3972.516, 4062.1304, 4152.896, 4244.804, 4337.8477, 4432.0176, 4527.304, 4623.6987, 4721.1914,
          4819.771, 4919.4272, 5020.1494, 5121.926, 5224.7446, 5328.5938]


@jit(nopython=True)
def vertical_interp(s_rho, variable, depths, mask_indices, H):
    t, k, j, i = variable.shape
    dst = np.full((t, len(depths), j, i), 1e37)

    for i, j in zip(*mask_indices):
        vertical_profile = variable[0, :, i, j][::-1]
        z_levels = H[i, j] * -s_rho[::-1]

        idx = (np.abs(np.array(depths) - z_levels[-1])).argmin()
        target_z_levels = depths[:idx + 1]
        target_z_levels = target_z_levels

        vertical_profile_interp = np.interp(target_z_levels, z_levels, vertical_profile)
        result = np.full(len(depths), np.nan)
        result[:len(vertical_profile_interp)] = vertical_profile_interp
        dst[0, :, i, j] = result

    return dst


@jit(nopython=True)
def extract_value_at_bottom(invar3d, invalid_value=1e37):
    t, k, lon, lat = invar3d.shape
    output2D = np.full((t, lon, lat), invalid_value)

    for i in range(lat):
        for j in range(lon):            
            vertical_profile = invar3d[0, :, j, i]
            valid_values = vertical_profile[vertical_profile != invalid_value]
        
            if len(valid_values) > 0:
                output2D[0, j, i] = valid_values[-1]
    
    return output2D


@jit(nopython=True)
def extract_value_at_surface(invar3d, factor=1.0, invalid_value=1e37):
    t, k, lon, lat = invar3d.shape
    output2D = np.full((lon, lat), invalid_value)

    for i in range(lat):
        for j in range(lon):
            val = invar3d[0, 0, j, i]
            if val != invalid_value:
                output2D[j, i] = val * factor
    
    return output2D


class Interp2D:
    def __init__(self, srcLons, srcLats, dstLons, dstLats):
        self.srcLons = srcLons
        self.srcLats = srcLats
        self.dstLons = dstLons
        self.dstLats = dstLats

    def interp(self, invar2d, fill_value=1.e+37, invalid_value=1.e+37):
        py = self.srcLats.flatten()
        px = self.srcLons.flatten()
        z = np.array(invar2d).flatten()
        if invalid_value is not None and not np.isnan(invalid_value):
            z[z == invalid_value] = np.nan
        X, Y = np.meshgrid(self.dstLons, self.dstLats)
        outvar2d = griddata((px, py), z, (X, Y), method='nearest', fill_value=fill_value)
        outvar2d[np.isnan(outvar2d)] = fill_value
        return outvar2d


class Interp3D(Interp2D):
    def __init__(self, srcLons, srcLats, dstLons, dstLats, s_rho, mask, H):
        super().__init__(srcLons, srcLats, dstLons, dstLats)
        self.s_rho = s_rho
        self.H = H
        self.mask = super().interp(mask, fill_value=0, invalid_value=np.nan)
        self.mask_indices = np.where(self.mask == 1)

    def interp(self, invar3d, fill_value=1.e+37, invalid_value=1.e+37):
        outvar3d = np.empty((invar3d.shape[0], invar3d.shape[1], len(self.dstLats), len(self.dstLons)))
        for k in range(invar3d.shape[1]):
            print(f"Interpolating level: {k}")
            outvar3d[0, k] = super().interp(invar3d[0, k])

        outvar3dZeta = vertical_interp(self.s_rho.filled(np.nan), outvar3d, depths, self.mask_indices, self.H.filled(np.nan))
        outvar3dZeta[np.isnan(outvar3dZeta)] = fill_value
        return outvar3dZeta

    def bottomValues(self, invar3d, invalid_value=1e37):
        return extract_value_at_bottom(invar3d, invalid_value)
    
    def surfaceValues(self, invar3d, factor=1.0, invalid_value=1e37):
        return extract_value_at_surface(invar3d, factor, invalid_value)
