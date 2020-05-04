"""
Same as backtrack_qss_figsrc but everything now based on lwc/meanr/nconc
calculated including rain as well as cloud drops.
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
versionstr = 'v6_'

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
    For both polluted and unpolluted model runs, plot some quantity described
    in the notes.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_ss_v4', 'r')
        ncsecvars = ncsecfile.variables
        nctertfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_ventilation_data', 'r')
        nctertvars = ncsecfile.variables
        
        #get relevant primary variables from wrf output
        SS = ncprimvars['SSW'][...]

        #get secondary variables
        lh_K_s = ncsecvars['lh_K_s'][...]
        lwc = ncsecvars['lwc_cloud'][...]
        ss_ry = ncsecvars['ss_ry'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        #get tertiary variables
        f = nctertvars['f'][...]

        #ventilation correction
        ss_ry = ss_ry/f
        
        ncprimfile.close()
        ncsecfile.close()
        nctertfile.close()
        
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
                                    (np.abs(w) > 2)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (lh_K_s > 0)))
        
        print(np.shape(mask))
        print('num above lwc cutoff: ', np.sum(mask))
        
        #do regression analysis
        m, b, R, sig = linregress(ss_ry[mask]*100, SS[mask]*100)
        print(m, b, R**2)
        
        #count number of points outside range [-100, 100] for qss_ry set
        n_hi = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 1), \
                                     mask)))
        n_lo = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < -1), \
                                     mask)))
        print(model_label)
        print('n_hi: ', n_hi)
        print('n_lo: ', n_lo)
        print('max ssry:', np.nanmax(ss_ry))
        print('min ssry:', np.nanmin(ss_ry))
        
        n_q1 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 0), \
                                    (SS > 0), \
                                     mask)))
        n_q2 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < 0), \
                                    (SS > 0), \
                                     mask)))
        n_q3 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < 0), \
                                    (SS < 0), \
                                     mask)))
        n_q4 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 0), \
                                    (SS < 0), \
                                     mask)))
        
        print('Number of points in Q1:', n_q1)
        print('Number of points in Q2:', n_q2)
        print('Number of points in Q3:', n_q3)
        print('Number of points in Q4:', n_q4)
        print()
        
        ##get limits of the data for plotting purposes
        #xlim_max = np.max(np.array( \
        #                [np.max(SS_qss[mask]), \
        #                 np.max(SS_wrf[mask])]))
        #xlim_min = np.min(np.array( \
        #                [np.min(SS_qss[mask]), \
        #                 np.min(SS_wrf[mask])]))
        #ax_lims = np.array([100*xlim_min, 100*xlim_max])
        #print(ax_lims)
        
        ax_lims = np.array([-100, 100])
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(ss_ry[mask]*100, SS[mask]*100, c=colors['ss'])
        ax.plot(ax_lims, np.add(b, m*ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xlabel(r'$SS_{RY}$ (%)')
        ax.set_ylabel(r'$SS_{WRF}$ (%)')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_backtrack_qss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
