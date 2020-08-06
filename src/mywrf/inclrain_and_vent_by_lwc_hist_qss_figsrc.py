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
versionstr = 'v1_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'qss': '#095793', 'wrf': '#BA3F00'}

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
        #pres = ncsecvars['pres'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncsecfile.close()

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        #e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau (B is slightly different
        #from Q_2 because)
        #B = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))
        #F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        #F_d = rho_w*R_v*temp/(D*e_s)
        
        #factor in denominator of Rogers and Yau qss ss formula (p 110)
        #denom = B/(F_k + F_d)

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        ss_qss = w*A/(4*np.pi*D*nconc*meanfr)
        
        lwc_e_5_cutoffs = [1., 2., 5., 10., 20., 50., 100.]

        for lwc_cutoff in lwc_e_5_cutoffs:
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
                                        (lwc > lwc_cutoff*10.**-5), \
                                        (temp > 273), \
                                        (w > 2)))
            #mask = np.logical_and.reduce(( \
            #                            (lwc > lwc_cutoff), \
            #                            (temp > 273), \
            #                            (np.abs(w) > 4)))
            #mask = np.logical_and.reduce(( \
            #                            (lwc > lwc_cutoff), \
            #                            (temp > 273), \
            #                            (w > 4)))
            
            fig, ax = plt.subplots()
            ax.hist(ss_qss[mask]*100, bins=30, color=colors['qss'], \
                    histtype='bar', log=True, label=r'$SS_{QSS}$')
            ax.hist(ss_wrf[mask]*100, bins=30, color=colors['wrf'], \
                    histtype='step', log=True, label=r'$SS_{WRF}$')
            ax.set_xlabel('SS (%)')
            ax.set_ylabel('log(Frequency)')
            ax.set_title('LWC > ' + str(lwc_cutoff) + '*10^-5')
            fig.set_size_inches(21, 12)

            outfile = FIG_DIR + versionstr + 'inclrain_and_vent_by_lwc_hist_qss_' \
                        + str(lwc_cutoff) + '_' + model_label + '_figure.png'
            plt.savefig(outfile)
            plt.close(fig=fig)

if __name__ == "__main__":
    main()
