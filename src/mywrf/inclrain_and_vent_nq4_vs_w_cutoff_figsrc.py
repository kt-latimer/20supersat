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
lwc_cutoff = 1.e-5
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
                            #'wrfout_d01_secondary_vars_no_rain', 'r')
        ncsecvars = ncsecfile.variables

        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        #meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        #pres = ncsecvars['pres'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]
        ncsecfile.close()

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        ss_qss = w*A/(4*np.pi*D*nconc*meanfr)

        del A, meanfr, nconc #for memory
        w_cutoffs = [i for i in range(2, 11)]
        nq4 = []
        ntot = []
        for w_cutoff in w_cutoffs:
            mask = np.logical_and.reduce(( \
                                        (lwc > lwc_cutoff), \
                                        (temp > 273), \
                                        (np.abs(w) > w_cutoff)))
            ss_qss_cutoff = ss_qss[mask]
            ss_wrf_cutoff = ss_wrf[mask]
            ntot.append(np.sum(mask))
            nq4.append(np.sum(np.logical_and(ss_qss_cutoff > 0, \
                                            ss_wrf_cutoff < 0)))
        
        #plot shit 
        fig, ax = plt.subplots()
        ax.scatter(w_cutoffs, np.array(nq4)/np.array(ntot)*100, c=colors['ss'])
        ax.set_xlabel(r'$w_{cutoff}$ (m/s)')
        ax.set_ylabel(r'$N_{Q4}/N_{tot}$ (%)')
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_nq4_vs_w_cutoff_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)
        
        del lwc, mask, ss_qss, ss_wrf, temp, w #for memory

if __name__ == "__main__":
    main()
