"""
plot constituent quantities of ss_qss vs ss_wrf to try to understand "two
prong" appearance of ss_wrf vs ss_qss plot
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
versionstr = 'v4_'

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
    For both polluted and unpolluted model runs, plot constituent quantities of
    qss supersat vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables

        #get secondary variables
        lh_K_s = ncsecvars['lh_K_s'][...]
        #lh_J_m3_s = ncsecvars['lh_J_m3_s'][...]
        lwc = ncsecvars['lwc_cloud'][...]
        pres = ncsecvars['pres'][...]
        rho_air = ncsecvars['rho_air'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        #ncprimfile.close()
        ncsecfile.close()

        #make filter mask
        #mask = LWC_C > lwc_cutoff
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (nconc > 3.e6)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1), \
        #                            (np.abs(w) < 10)))
        mask = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2)))
        #mask = np.logical_and(mask, ss_wrf < 0)
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
       
        lh_K_s = lh_K_s[mask]
        #lh_J_m3_s = lh_J_m3_s[mask]
        rho_air = rho_air[mask]
        pres = pres[mask]
        temp = temp[mask]
        w = w[mask]

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        Q_2 = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp

        qty1 = A*w/Q_2
        qty2 = C_ap*rho_air/(L_v*rho_w)*lh_K_s

        #do regression analysis
        m, b, R, sig = linregress(qty1, qty2)
        print(m, b, R**2)
        
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(qty1, qty2, c=colors['ss'])
        ax.set_xlim(np.array([np.min(qty1), np.max(qty1)]))
        ax.set_ylim(np.array([np.min(qty2), np.max(qty2)]))
        #ax.set_ylim(b + m*np.array([np.min(qty1), np.max(qty1)]))
        ax.plot(np.array(ax.get_xlim()), np.add(b, m*np.array(ax.get_xlim())), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        #ax.set_xlabel('LH (K/s)')
        ax.set_xlabel('qty1')
        ax.set_ylabel('qty2')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'fancy_w_vs_lh_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
