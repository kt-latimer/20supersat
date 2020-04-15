"""
Count points lying outside specified ranges in backtrack_qss analysis (fairly
self-explanatory from the printed output)
"""
from datetime import datetime

from netCDF4 import Dataset, MFDataset
import numpy as np

from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
versionstr = 'v1_'

#physical constants
g = 9.8 #grav accel (m/s^2)

def main():
    """
    main routine.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir + 'wrfout_d01_secondary_vars', 'r')
        ncsecvars = ncsecfile.variables
        
        #get relevant primary variables from wrf output
        LWC = ncprimvars['QCLOUD'][...]
        PH = ncprimvars['PH'][...] #geopotential perturbation
        PHB = ncprimvars['PHB'][...] #geopotential base value

        #get secondary variables
        meanr = ncsecvars['meanr'][...]

        ncprimfile.close()
        ncsecfile.close()

        #get altitude (geopotential) and realign to mass grid
        z = (PH + PHB)/g #altitude rel to sea level
        del PH, PHB
        z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2

        print(model_label)
        print('Total size: ', z.size)
        mask = LWC > 1.e-5 
        print('LWC above 1.e-5: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    (meanr > 0), \
                                    (meanr < 60.e-6)))
        print('LWC above 1.e-5 and meanr in [0, 60um]: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    z < 6000))
        print('LWC above 1.e-5 and meanr in [0, 60um] and alt below 6km: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    z < 3000))
        print('LWC above 1.e-5 and meanr in [0, 60um] and alt below 3km: ', np.sum(mask))
        mask = LWC > 5.e-5 
        print('LWC above 5.e-5: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    (meanr > 0), \
                                    (meanr < 60.e-6)))
        print('LWC above 5.e-5 and meanr in [0, 60um]: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    z < 6000))
        print('LWC above 5.e-5 and meanr in [0, 60um] and alt below 6km: ', np.sum(mask))
        mask = np.logical_and.reduce(( \
                                     mask, \
                                    z < 3000))
        print('LWC above 5.e-5 and meanr in [0, 60um] and alt below 3km: ', np.sum(mask))
        del LWC, mask, meanr, z

if __name__ == "__main__":
    main()
