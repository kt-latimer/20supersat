"""
compare saturation vapor pressure formulae
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

#coefficients in sat vap pres polynomial approx (Khain and Pinsky p 72)
a = [6.1117675e2, 4.43986062e1, 1.43053301, 2.65027242e-2, 3.02246994e-4,
        2.03886313e-6, 6.38780966e-9]

def main():
    """
    For both polluted and unpolluted model runs, plot e_sat versions against
    each other
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
        temp = ncsecvars['temp'][...]
        
        #convert to celcius
        temp_C = temp - 273.15

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to Pa (p 16)
        e_sat_1 = 611.2*np.exp(17.67*temp_C/(temp_C + 243.5))
        e_sat_2 = np.zeros(np.shape(e_sat_1))
        for i in range(7):
            e_sat_2 += a[i]*(temp_C**i)

        #make filter mask
        mask = LWC > lwc_cutoff

        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        im = ax.scatter(e_sat_1[mask]*100, e_sat_2[mask]*100, c=temp_C[mask], cmap='coolwarm')
        ax.set_xlabel('e_sat - RY (Pa)')
        ax.set_ylabel('e_sat - Flatau et al (Pa)')
        fig.set_size_inches(21, 12)
        fig.colorbar(im, ax=ax, label='Temperature (C)')
        
        outfile = FIG_DIR + versionstr + 'compare_esat_formulae' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
