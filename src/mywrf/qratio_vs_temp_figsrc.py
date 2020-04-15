"""
looking at temperature dependence of Q1/Q2 as defined in Rogers and Yau p 106.
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-5
versionstr = 'v1_'

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

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
    For both polluted and unpolluted model runs, plot Q1/Q2 vs T
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
        
        #get secondary variables
        pres = ncsecvars['pres'][...]
        rho_air = ncsecvars['rho_air'][...]
        temp = ncsecvars['temp'][...]
        
        #convert to celcius
        temp_C = temp - 273.15

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to Pa (p 16)
        e_sat = 611.2*np.exp(17.67*temp_C/(temp_C + 243.5))

        #quantities Q1 and Q2 def'd on p 106 of RY
        Q1 = 1./temp*(L_v*g/(R_v*C_ap*temp) - g/R_a)
        Q2 = rho_air*(R_v*temp/e_sat + R_a*L_v**2./(R_v*pres*temp*C_ap))
        qratio = Q1/Q2
        print(np.nanmin(qratio))
        print(np.nanmax(qratio))
        print(np.nanmean(qratio))
        print(np.nanmedian(qratio))
        print(np.nanstd(qratio))

        return 

        #make filter mask
        mask = LWC > lwc_cutoff

        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        im = ax.scatter(temp[mask], qratio[mask])
        ax.set_xlabel('T (K)')
        ax.set_ylabel('Q1/Q2')
        fig.set_size_inches(21, 12)
        
        outfile = FIG_DIR + versionstr + 'qratio_vs_temp_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
