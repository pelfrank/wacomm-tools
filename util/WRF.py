from netCDF4 import Dataset, date2num

class WRF:
    def __init__(self, filename, time, lons, lats):
        self.lons = lons
        self.lats = lats
        self.time = time

        self.dwspd10 = None
        self.dwdir10 = None
        self.drain = None
        self.hrain = None
        self.hswe = None
        self.pw = None
        self.rh2 = None
        self.t2c = None
        self.uh = None
        self.srh = None
        self.mcape = None
        self.mcin = None
        self.u1000 = None
        self.v1000 = None
        self.tc1000 = None
        self.rh1000 = None
        self.u975 = None
        self.v975 = None
        self.tc975 = None
        self.rh975 = None
        self.u950 = None
        self.v950 = None
        self.tc950 = None
        self.rh950 = None
        self.u925 = None
        self.v925 = None
        self.tc925 = None
        self.rh925 = None
        self.u850 = None
        self.v850 = None
        self.tc850 = None
        self.rh850 = None
        self.u700 = None
        self.v700 = None
        self.tc700 = None
        self.rh700 = None
        self.u500 = None
        self.v500 = None
        self.tc500 = None
        self.rh500 = None
        self.u300 = None
        self.v300 = None
        self.tc300 = None
        self.rh300 = None
        self.tt = None
        self.ki = None
        self.theta_e850 = None
        self.theta_w850 = None
        self.delta_theta = None
        self.gph500 = None
        self.gph850 = None
        self.slp = None
        self.clf = None
        self.u10m = None
        self.v10m = None
        self.wspd10 = None
        self.wdir10 = None

        self.ncdstfile = Dataset(filename, "w", format="NETCDF4")

        self.ncdstfile.createDimension("time", size=1)
        self.ncdstfile.createDimension("latitude", size=len(lats))
        self.ncdstfile.createDimension("longitude", size=len(lons))

        self.timeVar = self.ncdstfile.createVariable("time", "i4", "time")
        self.timeVar.description = "Time"
        self.timeVar.long_name = "time"
        self.timeVar.units = "hours since 1900-01-01 00:00:0.0"

        self.lonVar = self.ncdstfile.createVariable("longitude", "f4", "longitude")
        self.lonVar.description = "Longitude"
        self.lonVar.long_name = "longitude"
        self.lonVar.units = "degrees_east"

        self.latVar = self.ncdstfile.createVariable("latitude", "f4", "latitude")
        self.latVar.description = "Latitude"
        self.lonVar.long_name = "latitude"
        self.latVar.units = "degrees_north"

        self.hsweVar = self.ncdstfile.createVariable("HOURLY_SWE", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.hsweVar.description = "Snow water equivalent"
        self.hsweVar.units = "kg m-2"

        self.hrainVar = self.ncdstfile.createVariable("DELTA_RAIN", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.hrainVar.description = "Hourly cumulated rain"
        self.hrainVar.units = "mm"

        self.drainVar = self.ncdstfile.createVariable("DAILY_RAIN", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.drainVar.description = "Daily cumulated rain"
        self.drainVar.units = "mm"

        self.t2cVar = self.ncdstfile.createVariable("T2C", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.t2cVar.description = "Temperature at 2m in Celsius"
        self.t2cVar.units = "C"

        self.rh2Var = self.ncdstfile.createVariable("RH2", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh2Var.description = "Relative humidity at 2 meters"
        self.rh2Var.units = "%"

        self.pwVar = self.ncdstfile.createVariable("PW", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.pwVar.description = "Precipitable Water"
        self.pwVar.units = "kg m-2"

        self.uhVar = self.ncdstfile.createVariable("UH", "f4", ("time", "latitude", "longitude") ,fill_value=1.e+37, zlib=True, complevel=4)
        self.uhVar.description = "Updraft Helicity"
        self.uhVar.units = "m2 s-2"

        self.srhVar = self.ncdstfile.createVariable("SRH", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.srhVar.description = "Storm Relative Helicity"
        self.srhVar.units = "m2 s-2"

        self.mcapeVar = self.ncdstfile.createVariable("MCAPE", "f4", ("time", "latitude", "longitude"), fill_value=9.96921e+36, zlib=True, complevel=4)
        self.mcapeVar.description = "Most unstable convective available potential energy"
        self.mcapeVar.units = "J kg-1"

        self.mcinVar = self.ncdstfile.createVariable("MCIN", "f4", ("time", "latitude", "longitude"), fill_value=9.96921e+36, zlib=True, complevel=4)
        self.mcinVar.description = "Maximum convective inibition"
        self.mcinVar.units = "J kg-1"

        self.u1000Var = self.ncdstfile.createVariable("U1000", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u1000Var.description = "grid rel. x-wind component at 1000 HPa"
        self.u1000Var.standard_name = "u-component"
        self.u1000Var.units = "m s-1"

        self.v1000Var = self.ncdstfile.createVariable("V1000", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v1000Var.description = "grid rel. y-wind component at 1000 HPa"
        self.v1000Var.standard_name = "v-component"
        self.v1000Var.units = "m s-1"

        self.tc1000Var = self.ncdstfile.createVariable("TC1000", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc1000Var.description = "Temperature at 1000 HPa"
        self.tc1000Var.units = "C"

        self.rh1000Var = self.ncdstfile.createVariable("RH1000", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh1000Var.description = "Relative humidity at 1000 HPa"
        self.rh1000Var.units = "%"

        self.u975Var = self.ncdstfile.createVariable("U975", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u975Var.description = "grid rel. x-wind component at 975 HPa"
        self.u975Var.standard_name = "u-component"
        self.u975Var.units = "m s-1"

        self.v975Var = self.ncdstfile.createVariable("V975", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v975Var.description = "grid rel. y-wind component at 975 HPa"
        self.v975Var.standard_name = "v-component"
        self.v975Var.units = "m s-1"

        self.tc975Var = self.ncdstfile.createVariable("TC975", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc975Var.description="Temperature at 975 HPa"
        self.tc975Var.units = "C"

        self.rh975Var = self.ncdstfile.createVariable("RH975", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh975Var.description = "Relative humidity at 975 HPa"
        self.rh975Var.units = "%"

        self.u950Var = self.ncdstfile.createVariable("U950", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u950Var.description = "grid rel. x-wind component at 950 HPa"
        self.u950Var.standard_name = "u-component"
        self.u950Var.units = "m s-1"

        self.v950Var = self.ncdstfile.createVariable("V950", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v950Var.description = "grid rel. y-wind component at 950 HPa"
        self.v950Var.standard_name = "v-component"
        self.v950Var.units = "m s-1"

        self.tc950Var = self.ncdstfile.createVariable("TC950", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc950Var.description = "Temperature at 950 HPa"
        self.tc950Var.units = "C"

        self.rh950Var = self.ncdstfile.createVariable("RH950", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh950Var.description = "Relative humidity at 950 HPa"
        self.rh950Var.units = "%"

        self.u925Var = self.ncdstfile.createVariable("U925", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u925Var.description = "grid rel. x-wind component at 925 HPa"
        self.u925Var.standard_name = "u-component"
        self.u925Var.units = "m s-1"

        self.v925Var = self.ncdstfile.createVariable("V925", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v925Var.description = "grid rel. y-wind component at 925 HPa"
        self.v925Var.standard_name = "v-component"
        self.v925Var.units = "m s-1"

        self.tc925Var = self.ncdstfile.createVariable("TC925", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc925Var.description = "Temperature at 925 HPa"
        self.tc925Var.units = "C"

        self.rh925Var = self.ncdstfile.createVariable("RH925", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh925Var.description = "Relative humidity at 925 HPa"
        self.rh925Var.units = "%"

        self.u850Var = self.ncdstfile.createVariable("U850", "f4",("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u850Var.description = "grid rel. x-wind component at 850 HPa"
        self.u850Var.standard_name = "u-component"
        self.u850Var.units = "m s-1"

        self.v850Var = self.ncdstfile.createVariable("V850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v850Var.description = "grid rel. y-wind component at 850 HPa"
        self.v850Var.standard_name = "v-component"
        self.v850Var.units = "m s-1"

        self.kiVar = self.ncdstfile.createVariable("KI", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.kiVar.description = "K-Index"
        self.kiVar.units = "C"

        self.ttVar =self. ncdstfile.createVariable("TT", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.ttVar.description = "Total Totals index"
        self.ttVar.units = "C"

        self.tc850Var = self.ncdstfile.createVariable("TC850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc850Var.description = "Temperature at 850 HPa"
        self.tc850Var.units = "C"

        self.theta_e850Var = self.ncdstfile.createVariable("THETA_E850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.theta_e850Var.description = "Equivalent Potential Temperature at 850 HPa"
        self.theta_e850Var.units = "C"

        self.theta_w850Var = self.ncdstfile.createVariable("THETA_W850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.theta_w850Var.description = "Wet Bulb Temperature at 850 HPa"
        self.theta_w850Var.units = "C"

        self.delta_thetaVar = self.ncdstfile.createVariable("DELTA_THETA", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.delta_thetaVar.description = "Differnce between Equivalent Potential Temperature at 500 HPa and at 850 HPa"
        self.delta_thetaVar.units = "C"

        self.rh850Var = self.ncdstfile.createVariable("RH850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh850Var.description = "Relative humidity at 850 HPa"
        self.rh850Var.units = "%"

        self.u700Var = self.ncdstfile.createVariable("U700", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u700Var.description = "grid rel. x-wind component at 700 HPa"
        self.u700Var.standard_name = "u-component"
        self.u700Var.units = "m s-1"

        self.v700Var = self.ncdstfile.createVariable("V700", "f4", ("time","latitude","longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v700Var.description = "grid rel. y-wind component at 700 HPa"
        self.v700Var.standard_name = "v-component"
        self.v700Var.units = "m s-1"

        self.tc700Var = self.ncdstfile.createVariable("TC700", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc700Var.description = "Temperature at 700 HPa"
        self.tc700Var.units = "C"

        self.rh700Var = self.ncdstfile.createVariable("RH700", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh700Var.description = "Relative humidity at 700 HPa"
        self.rh700Var.units = "%"

        self.u500Var = self.ncdstfile.createVariable("U500", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u500Var.description = "grid rel. x-wind component at 500 HPa"
        self.u500Var.standard_name = "u-component"
        self.u500Var.units = "m s-1"

        self.v500Var = self.ncdstfile.createVariable("V500", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v500Var.description = "grid rel. y-wind component at 500 HPa"
        self.v500Var.standard_name = "v-component"
        self.v500Var.units = "m s-1"

        self.tc500Var = self.ncdstfile.createVariable("TC500", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc500Var.description = "Temperature at 500 HPa"
        self.tc500Var.units = "C"

        self.rh500Var = self.ncdstfile.createVariable("RH500", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh500Var.description = "Relative humidity at 500 HPa"
        self.rh500Var.units = "%"

        self.u300Var = self.ncdstfile.createVariable("U300", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u300Var.description = "grid rel. x-wind component at 300 HPa"
        self.u300Var.standard_name = "u-component"
        self.u300Var.units = "m s-1"

        self.v300Var = self.ncdstfile.createVariable("V300", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v300Var.description = "grid rel. y-wind component at 300 HPa"
        self.v300Var.standard_name = "v-component"
        self.v300Var.units = "m s-1"

        self.tc300Var = self.ncdstfile.createVariable("TC300", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.tc300Var.description = "Temperature at 300 HPa"
        self.tc300Var.units = "C"

        self.rh300Var = self.ncdstfile.createVariable("RH300", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.rh300Var.description = "Relative humidity at 300 HPa"
        self.rh300Var.units = "%"

        self.gph500Var = self.ncdstfile.createVariable("GPH500", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.gph500Var.description = "Geopotential height at 500 HPa"
        self.gph500Var.units = "dm"

        self.gph850Var = self.ncdstfile.createVariable("GPH850", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.gph850Var.description = "Geopotential height at 850 HPa"
        self.gph850Var.units = "dm"

        self.slpVar = self.ncdstfile.createVariable("SLP", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.slpVar.description = "Sea level pressure"
        self.slpVar.units = "HPa"

        self.clfVar = self.ncdstfile.createVariable("CLDFRA_TOTAL", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.clfVar.description = "Total cloud fraction"
        self.clfVar.units = "%"

        self.u10mVar = self.ncdstfile.createVariable("U10M", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.u10mVar.description = "grid rel. x-wind component"
        self.u10mVar.standard_name = "u-component"
        self.u10mVar.units = "m s-1"

        self.v10mVar = self.ncdstfile.createVariable("V10M", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.v10mVar.description = "grid rel. y-wind component"
        self.v10mVar.standard_name = "v-component"
        self.v10mVar.units = "m s-1"

        self.wspd10Var = self.ncdstfile.createVariable("WSPD10", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.wspd10Var.description = "wind speed at 10 meters"
        self.wspd10Var.units = "m s-1"
        self.wspd10Var.standard_name = ""

        self.wdir10Var = self.ncdstfile.createVariable("WDIR10", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.wdir10Var.description = "wind dir at 10 meters"
        self.wdir10Var.units = "nord degrees"
        self.wdir10Var.standard_name = ""

        self.dwspd10Var = self.ncdstfile.createVariable("DELTA_WSPD10", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.dwspd10Var.description = "Difference of wind speed at 10 meters"
        self.dwspd10Var.units = "m s-1"
        self.dwspd10Var.standard_name = ""

        self.dwdir10Var = self.ncdstfile.createVariable("DELTA_WDIR10", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37, zlib=True, complevel=4)
        self.dwdir10Var.description = "Difference of wind dir at 10 meters"
        self.dwdir10Var.units = "nord degrees"
        self.dwdir10Var.standard_name = ""

    def write(self):
        self.timeVar[:] = date2num(self.time, units = self.timeVar.units)
        self.lonVar[:] = self.lons
        self.latVar[:] = self.lats
        
        self.dwspd10Var[:] = self.dwspd10
        self.dwdir10Var[:] = self.dwdir10
        self.drainVar[:] = self.drain
        self.hrainVar[:] = self.hrain
        self.hsweVar[:] = self.hswe
        self.pwVar[:] = self.pw
        self.rh2Var[:] = self.rh2
        self.t2cVar[:] = self.t2c
        self.uhVar[:] = self.uh
        self.srhVar[:] = self.srh
        self.mcapeVar[:] = self.mcape
        self.mcinVar[:] = self.mcin
        self.u1000Var[:] = self.u1000
        self.v1000Var[:] = self.v1000
        self.tc1000Var[:] = self.tc1000
        self.rh1000Var[:] = self.rh1000
        self.u975Var[:] = self.u975
        self.v975Var[:] = self.v975
        self.tc975Var[:] = self.tc975
        self.rh975Var[:] = self.rh975
        self.u950Var[:] = self.u950
        self.v950Var[:] = self.v950
        self.tc950Var[:] = self.tc950
        self.rh950Var[:] = self.rh950
        self.u925Var[:] = self.u925
        self.v925Var[:] = self.v925
        self.tc925Var[:] = self.tc925
        self.rh925Var[:] = self.rh925
        self.u850Var[:] = self.u850
        self.v850Var[:] = self.v850
        self.tc850Var[:] = self.tc850
        self.rh850Var[:] = self.rh850
        self.u700Var[:] = self.u700
        self.v700Var[:] = self.v700
        self.tc700Var[:] = self.tc700
        self.rh700Var[:] = self.rh700
        self.u500Var[:] = self.u500
        self.v500Var[:] = self.v500
        self.tc500Var[:] = self.tc500
        self.rh500Var[:] = self.rh500
        self.u300Var[:] = self.u300
        self.v300Var[:] = self.v300
        self.tc300Var[:] = self.tc300
        self.rh300Var[:] = self.rh300
        self.ttVar[:] = self.tt
        self.kiVar[:] = self.ki
        self.theta_e850Var[:] = self.theta_e850
        self.theta_w850Var[:] = self.theta_w850
        self.delta_thetaVar[:] = self.delta_theta
        self.gph500Var[:] = self.gph500
        self.gph850Var[:] = self.gph850
        self.slpVar[:] = self.slp
        self.clfVar[:] = self.clf
        self.u10mVar[:] = self.u10m
        self.v10mVar[:] = self.v10m
        self.wspd10Var[:] = self.wspd10
        self.wdir10Var[:] = self.wdir10

    def close(self):
        if self.ncdstfile:
            self.ncdstfile.close()
