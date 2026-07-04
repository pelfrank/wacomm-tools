from netCDF4 import Dataset


class WW33:
    def __init__(self, filename, time, lons, lats):
        self.lons = lons
        self.lats = lats
        self.time = time

        self.dpt = None
        self.hs = None
        self.lm = None
        self.fp = None
        self.dir = None
        self.period = None

        self.ncdstfile = Dataset(filename, "w", format="NETCDF4")

        self.ncdstfile.createDimension("time", size=1)
        self.ncdstfile.createDimension("latitude", size=len(lats))
        self.ncdstfile.createDimension("longitude", size=len(lons))

        self.timeVar = self.ncdstfile.createVariable("time", "f4", "time")
        self.timeVar.description = "Time since initialization"
        self.timeVar.field = "time, scalar, series"
        self.timeVar.long_name = "julian day (UT)"
        self.timeVar.standard_name = "time" 
        self.timeVar.calendar = "standard" 
        self.timeVar.units = "days since 1990-01-01 00:00:00" 
        self.timeVar.conventions = "relative julian days with decimal part (as parts of the day )" 
        self.timeVar.axis = "T"

        self.lonVar = self.ncdstfile.createVariable("longitude", "f4", "longitude")
        self.lonVar.description = "Longitude"
        self.lonVar.long_name = "longitude"
        self.lonVar.units = "degrees_east"
        self.lonVar.standard_name = "longitude" 
        self.lonVar.valid_min = -180.0
        self.lonVar.valid_max = 180.0
        self.lonVar.axis = "X" 

        self.latVar = self.ncdstfile.createVariable("latitude", "f4", "latitude")
        self.latVar.description = "Latitude"
        self.lonVar.long_name = "latitude"
        self.latVar.units = "degrees_north"
        self.latVar.long_name = "latitude" 
        self.latVar.standard_name = "latitude" 
        self.latVar.valid_min = -90.0 
        self.latVar.valid_max = 90.0
        self.latVar.axis = "Y"

        self.dptVar = self.ncdstfile.createVariable("dpt", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.dptVar.description = "Depth"
        self.dptVar.long_name = "depth" 
        self.dptVar.standard_name = "depth" 
        self.dptVar.globwave_name = "depth"
        self.dptVar.units = "m"
        self.dptVar.scale_factor = 1.0
        self.dptVar.add_offset = 0.0
        self.dptVar.valid_min = -90000
        self.dptVar.valid_max = 140000

        self.hsVar = self.ncdstfile.createVariable("hs", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.hsVar.description = "Significant wave height"
        self.hsVar.long_name = "significant height of wind and swell waves" 
        self.hsVar.standard_name = "sea_surface_wave_significant_height" 
        self.hsVar.globwave_name = "significant_wave_height" 
        self.hsVar.units = "m" 
        self.hsVar.scale_factor = 1.0
        self.hsVar.add_offset = 0.0
        self.hsVar.valid_min = 0.0
        self.hsVar.valid_max = 100.0

        self.lmVar = self.ncdstfile.createVariable("lm", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.lmVar.description = "Mean weave length"
        self.lmVar.long_name = "mean wave length" 
        self.lmVar.standard_name = "mean_wave_length" 
        self.lmVar.globwave_name = "mean_wave_length" 
        self.lmVar.units = "m" 
        self.lmVar.scale_factor = 1.0
        self.lmVar.add_offset = 0.0
        self.lmVar.valid_min = 0 
        self.lmVar.valid_max = 3200 

        self.fpVar = self.ncdstfile.createVariable("fp", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.fpVar.description = "Wave peak frequency"
        self.fpVar.long_name = "wave peak frequency" 
        self.fpVar.standard_name = "sea_surface_wave_peak_frequency" 
        self.fpVar.globwave_name = "dominant_wave_frequency" 
        self.fpVar.units = "s-1" 
        self.fpVar.scale_factor = 1.0
        self.fpVar.add_offset = 0.0
        self.fpVar.valid_min = 0 
        self.fpVar.valid_max = 10000 

        self.dirVar = self.ncdstfile.createVariable("dir", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.dirVar.description = "Wave mean direction"
        self.dirVar.long_name = "wave mean direction" 
        self.dirVar.standard_name = "sea_surface_wave_from_direction" 
        self.dirVar.globwave_name = "wave_from_direction" 
        self.dirVar.units = "degree" 
        self.dirVar.scale_factor = 1.0
        self.dirVar.add_offset = 0.0
        self.dirVar.valid_min = 0 
        self.dirVar.valid_max = 3600 

        self.periodVar = self.ncdstfile.createVariable("period", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.periodVar.description = "Mean period"
        self.periodVar.long_name = "mean period T0m1"
        self.periodVar.standard_name = "sea_surface_wind_wave_mean_period_from_variance_spectral_density_inverse_frequency_moment"
        self.periodVar.globwave_name = "mean_period_t0m1"
        self.periodVar.units = "s"
        self.periodVar.scale_factor = 1.0
        self.periodVar.add_offset = 0.0
        self.periodVar.valid_min = 0
        self.periodVar.valid_max = 5000 

    def write(self):
        self.timeVar[:] = self.time
        self.lonVar[:] = self.lons
        self.latVar[:] = self.lats

        self.dptVar[:] = self.dpt
        self.hsVar[:] = self.hs
        self.lmVar[:] = self.lm
        self.fpVar[:] = self.fp
        self.dirVar[:] = self.dir
        self.periodVar[:] = self.period

    def close(self):
        if self.ncdstfile:
            self.ncdstfile.close()