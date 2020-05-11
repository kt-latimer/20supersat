"""
Identify points where wrf ss disagrees by at least a factor of 100
with qss / ry ss approximations. Print to SLURM output
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-5
versionstr = 'v1_'

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

def main():
    """
    For both polluted and unpolluted model runs, find outlier points, print
    indices and associated environmental data. 
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        ncsecfile = Dataset(DATA_DIR + model_dir +
                    'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        nconc = ncsecvars['nconc'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncsecfile.close()

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        ss_qss = w*A/(4*np.pi*D*nconc*meanfr)

        #make filter mask
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (w > 2), \
        #                            (meanr > 0), \
        #                            (meanr < 60.e-6)))
        mask2 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2), \
                                    (ss_qss > 0), \
                                    (ss_wrf < 0)))
        mask1 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2), \
                                    (ss_qss > 0), \
                                    (ss_wrf > 0)))

        print('lwc (super)sat updrafts mean: ', np.nanmean(lwc[mask1]))
        print('lwc (super)sat updrafts mean: ', np.nanmedian(lwc[mask1]))
        print('lwc (super)sat updrafts stdev: ', np.nanstd(lwc[mask1]))
        print('lwc (super)sat updrafts max: ', np.nanmax(lwc[mask1]))
        print('lwc (super)sat updrafts min: ', np.nanmin(lwc[mask1]))
        print('lwc subsat updrafts mean: ', np.nanmean(lwc[mask2]))
        print('lwc subsat updrafts median: ', np.nanmedian(lwc[mask2]))
        print('lwc subsat updrafts stdev: ', np.nanstd(lwc[mask2]))
        print('lwc subsat updrafts max: ', np.nanmax(lwc[mask2]))
        print('lwc subsat updrafts min: ', np.nanmin(lwc[mask2]))

if __name__ == "__main__":
    main()
