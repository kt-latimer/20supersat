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
versionstr = 'v3_'

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
        #ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        #ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables

        ##get primary variables
        #q_ice = ncprimvars['QICE'][...]
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        nconc = ncsecvars['nconc'][...]
        #nconccloud = ncsecvars['nconccloud'][...]
        #nconcrain = ncsecvars['nconcrain'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        #ncprimfile.close()
        ncsecfile.close()

        #lwc = lwc + q_ice

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        
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
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
       
        A = A[mask]
        meanfr = meanfr[mask]
        nconc = nconc[mask]
        #nconccloud = nconccloud[mask]
        #nconcrain = nconcrain[mask]
        ss_wrf = ss_wrf[mask]
        w = w[mask]

        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter(A, ss_wrf*100, c=colors['ss'])
        ax[0][1].scatter(meanfr, ss_wrf*100, c=colors['ss'])
        ax[1][0].scatter(nconc, ss_wrf*100, c=colors['ss'])
        ax[1][1].scatter(w, ss_wrf*100, c=colors['ss'])
        #ax[1][0].scatter(nconccloud, ss_wrf*100, c=colors['ss'])
        #ax[1][1].scatter(nconcrain, ss_wrf*100, c=colors['ss'])
        ax[0][0].set_xlabel(r'$A$ ()')
        ax[0][1].set_xlabel(r'$\langle f(r) \cdot r \rangle$ (m)')
        ax[1][0].set_xlabel('Num. Conc. (m^-3)')
        ax[1][1].set_xlabel('w (m/s)')
        #ax[1][0].set_xlabel('Num. Conc. [cloud] (m^-3)')
        #ax[1][1].set_xlabel('Num. Conc. [rain] (m^-3)')
        ax[0][0].set_ylabel(r'$SS_{WRF}$ (%)')
        ax[0][1].set_ylabel(r'$SS_{WRF}$ (%)')
        ax[1][0].set_ylabel(r'$SS_{WRF}$ (%)')
        ax[1][1].set_ylabel(r'$SS_{WRF}$ (%)')
        ax[0][0].set_xlim(np.array([-1.e-3, 1.e-3]))
        ax[0][1].set_xlim(np.array([-9.e-4, 9.e-4]))
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_breakdown_qss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        del A, lwc, mask, meanfr, nconc, ss_wrf, temp, w #for memory

if __name__ == "__main__":
    main()
