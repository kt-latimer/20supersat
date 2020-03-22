"""
Create and save figure qss_vs_wrf. This is a scatter plot comparing WRF's
supersaturation output against a simplified version of the quasi-steady-state
supersaturation equation in Korolev 2003.
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
versionstr = 'v5_'

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'denom': '#88720A'}

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
    For both polluted and unpolluted model runs, plot qss SS approx vs WRF SS.
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
        temp = ncsecvars['temp'][...]

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        Q_2 = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))
        F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        F_d = rho_w*R_v*temp/(D*e_s)

        #factor in denominator of Rogers and Yau qss ss formula (p 110)
        denom = Q_2/(F_k + F_d)

        #make mask on LWC values
        mask = LWC > lwc_cutoff
        #print(np.sum(mask))
        #print('denom stats')
        #print(np.nanmean(denom[mask]))
        #print(np.nanstd(denom[mask]))
        #print('Q_2 stats')
        #print(np.nanmean(Q_2[mask]))
        #print(np.nanstd(Q_2[mask]))
        #print('F_k stats')
        #print(np.nanmean(F_k[mask]))
        #print(np.nanstd(F_k[mask]))
        #print('F_d stats')
        #print(np.nanmean(F_d[mask]))
        #print(np.nanstd(F_d[mask]))
        #plot the supersaturations against each other with regression line
        #fig, ax = plt.subplots()
        #ax.scatter(denom[mask], D*np.ones(np.shape(denom))[mask], c=colors['denom'])
        #ax.set_xlabel('RY denominator')
        #ax.set_ylabel('constant denominator')
        #fig.set_size_inches(21, 12)
        fact = 1./denom
        plt.hist(fact[mask], bins=30)
        outfile = FIG_DIR + versionstr + 'qss_expand_denom_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)

if __name__ == "__main__":
    main()
