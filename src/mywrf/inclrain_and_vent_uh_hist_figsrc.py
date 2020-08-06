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
    For both polluted and unpolluted model runs, plot qss vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncprimvars = ncprimfile.variables
        ncsecvars = ncsecfile.variables
        
        #get primary variables
        U = ncprimvars['U'][...]
        V = ncprimvars['V'][...]
        
        #align to mass grid
        u = (U[:, :, :, 0:-1] + U[:, :, :, 1:])/2.
        v = (V[:, :, 0:-1, :] + V[:, :, 1:, :])/2.

        #get magnitude of horizontal velocity
        uh = np.sqrt(u**2. + v**2.)

        del u, v, U, V #for memory

        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncprimfile.close()
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
        mask1 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (ss_wrf > 0), \
                                    (w > 2)))
        mask2 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (ss_wrf < 0), \
                                    (w > 2)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
        
        ##get limits of the data for plotting purposes
        #xlim_max = np.max(np.array( \
        #                [np.max(ss_wrf_qss[mask]), \
        #                 np.max(ss_wrf_wrf[mask])]))
        #xlim_min = np.min(np.array( \
        #                [np.min(ss_wrf_qss[mask]), \
        #                 np.min(ss_wrf_wrf[mask])]))
        #ax_lims = np.array([100*xlim_min, 100*xlim_max])
        #print(ax_lims)
        
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.hist(uh[mask1]/w[mask1], bins = 40, label='Cloud bulk', density=True)
        ax.hist(uh[mask2]/w[mask2], bins = 40, label='Cloud edge', density=True)
        #ax.set_xlabel(r'$|u_h|$ (m/s)')
        ax.set_xlabel(r'$|u_h|/|w|$ (m/s)')
        ax.set_ylabel('Frequency')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_uh_hist_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
