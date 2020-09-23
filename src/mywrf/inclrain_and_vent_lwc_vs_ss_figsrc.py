"""
Same as inclrain_qss_vs_fan but with ventilation correction 
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-4
versionstr = 'v1_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

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
    For both polluted and unpolluted model runs, plot qss vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        nconc = ncsecvars['nconc'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        pres = ncsecvars['pres'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncsecfile.close()
        
        #calc full ss_qss
        rho_a = pres/(R_a*temp)
        del pres #for memory

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        e_s = get_sat_vap_pres(temp)
        B = rho_w*(R_v*temp/e_s + L_v**2./(R_v*C_ap*rho_a*temp**2.))
        del rho_a #for memory

        F_d = rho_w*R_v*temp/(D*e_s) 
        del e_s #for memory

        F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)
        ss_qss = A*w*(F_d + F_k)/(4*np.pi*nconc*meanfr*B)*100
        #A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        #ss_qss = w*A/(4*np.pi*D*nconc*meanfr)
        del A, B, F_d, F_k, meanfr, nconc #for memory

        #make filter mask
        mask = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2)))
        
        print(np.shape(mask))
        print('num above lwc cutoff: ', np.sum(mask))
        
        ss_qss = ss_qss[mask]
        lwc = lwc[mask]
        m, b, R, sig = linregress(lwc, ss_qss)

        xmin = np.min(lwc)
        xmax = np.max(lwc)

        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.scatter(w, ss_qss, alpha=0.4)
        ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
                color='k', linestyle='--', \
                label=('m = ' + str(np.round(m, decimals=2)) + \
                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_title('Liquid water content in clouds vs supersaturation, LWC > 1.e-4 g/g, T > 273K, w > 2 m/s')
        ax.set_xlabel('LWC (kg/kg)')
        ax.set_ylabel('SS (%)')
        ax.legend(loc=1)
        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_lwc_vs_ss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

def get_sat_vap_pres(T):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return e_s

if __name__ == "__main__":
    main()