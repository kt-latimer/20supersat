"""
Same as inclrain_cloudmap_figsrc, but specifically for east/west slice 201;
create one figure for each time step in the model output 
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
versionstr = 'v2_'

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
    For both polluted and unpolluted model runs, create and save heatmap 
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        
        print(model_label)

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables

        #get primary variables
        PH = ncprimvars['PH'][...] #geopotential perturbation
        PHB = ncprimvars['PHB'][...] #geopotential base value

        #get altitude (geopotential) NOT aligned to mass grid
        z = (PH + PHB)/g #altitude rel to sea level
        del PH, PHB
        
        for j in range(z.shape[1]):
            print(np.mean(z[:, j, :, :]), np.std(z[:, j, :, :]))

if __name__ == "__main__":
    main()
