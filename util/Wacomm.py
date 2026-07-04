from netCDF4 import Dataset


class Wacomm:
    def __init__(self, filename, time, depths, lons, lats):
        self.lons = lons
        self.lats = lats
        self.depths = depths
        self.time = time

        self.conc = None
        self.sfconc = None
        self.sfconc_10m = None
        self.sfconc_30m = None
        self.mask = None

        self.ncdstfile = Dataset(filename, "w", format="NETCDF4")

        self.ncdstfile.createDimension("time", size=1)
        self.ncdstfile.createDimension("depth", size=len(depths))
        self.ncdstfile.createDimension("latitude", size=len(lats))
        self.ncdstfile.createDimension("longitude", size=len(lons))

        self.timeVar = self.ncdstfile.createVariable("time", "i4", "time")
        self.timeVar.description = "Time since initialization"
        self.timeVar.long_name = "time since initialization"
        self.timeVar.units = "seconds since 1968-05-23 00:00:00"
        self.timeVar.calendar = "gregorian"
        self.timeVar.field = "time, scalar, series"
        self.timeVar.standard_name = "time"
        self.timeVar.axis = "T"

        self.depthVar = self.ncdstfile.createVariable("depth", "f4", "depth")
        self.depthVar.description = "depth"
        self.depthVar.long_name = "depth"
        self.depthVar.units = "meters"
        self.depthVar.standard_name = "depth"
        self.depthVar.axis = "Z"

        self.lonVar = self.ncdstfile.createVariable("longitude", "f4", "longitude")
        self.lonVar.description = "Longitude"
        self.lonVar.long_name = "longitude"
        self.lonVar.units = "degrees_east"
        self.lonVar.standard_name = "longitude"
        self.lonVar.axis = "X"

        self.latVar = self.ncdstfile.createVariable("latitude", "f4", "latitude")
        self.latVar.description = "Latitude"
        self.lonVar.long_name = "latitude"
        self.latVar.units = "degrees_north"
        self.latVar.standard_name = "latitude"
        self.latVar.axis = "Y"

        self.maskVar = self.ncdstfile.createVariable("mask", "f4", ("latitude", "longitude"), fill_value=1.e37, zlib=True, complevel=4)
        self.maskVar.option_0 = "land"
        self.maskVar.option_1 = "water"
        self.maskVar.long_name = "mask on RHO points"

        self.concVar = self.ncdstfile.createVariable("conc", "f4", ("time", "depth", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.concVar.description = "concentration of suspended matter in sea water"
        self.concVar.units = "1"
        self.concVar.long_name = "concentration"

        self.sfconcVar = self.ncdstfile.createVariable("sfconc", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.sfconcVar.description = "concentration of suspended matter at the surface"
        self.sfconcVar.units = "1"
        self.sfconcVar.long_name = "surface_concentration"

        self.sfconc10mVar = self.ncdstfile.createVariable("sfconc10m", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.sfconc10mVar.description = "concentration of suspended matter at the surface at 10 meters"
        self.sfconc10mVar.units = "1"
        self.sfconc10mVar.long_name = "surface_concentration_10m"

        self.sfconc30mVar = self.ncdstfile.createVariable("sfconc30m", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.sfconc30mVar.description = "concentration of suspended matter at the surface at 30 meters"
        self.sfconc30mVar.units = "1"
        self.sfconc30mVar.long_name = "surface_concentration_30m"

    def write(self):
        self.timeVar[:] = self.time
        self.lonVar[:] = self.lons
        self.latVar[:] = self.lats
        self.depthVar[:] = self.depths

        self.concVar[:] = self.conc
        self.sfconcVar[:] = self.sfconc
        self.sfconc10mVar[:] = self.sfconc_10m
        self.sfconc30mVar[:] = self.sfconc_30m
        self.maskVar[:] = self.mask

    def close(self):
        if self.ncdstfile:
            self.ncdstfile.close()
