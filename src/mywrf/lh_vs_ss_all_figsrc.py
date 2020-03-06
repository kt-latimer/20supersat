"""
Plot rate of latent heat release from water condensation/evaporation versus
supersaturation WRF variables and print metadata (i.e. point count in each
quadrant) to output file.
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

versionstr = 'v1_'

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'all': '#777777'}

def main():
    """
    for both C_BG and C_PI model configurations (see Fan 2018), plot LH rate vs
    SS. metadata output will be printed to file specified in slurm command.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncvars = ncfile.variables

        #get relevant variables from wrf output
        LH_wrf = ncvars['TEMPDIFFL'][...]
        SS_wrf = ncvars['SSW'][...]
        W_wrf = ncvars['W'][...]

        #get metadata
        n_q1 = np.sum(np.logical_and(LH_wrf > 0, SS_wrf > 0))
        n_q2 = np.sum(np.logical_and(LH_wrf < 0, SS_wrf > 0))
        n_q3 = np.sum(np.logical_and(LH_wrf < 0, SS_wrf < 0))
        n_q4 = np.sum(np.logical_and(LH_wrf > 0, SS_wrf < 0))

        #print metadata to output file
        print(model_label)
        print('Number of points in Q1:', n_q1)
        print('Number of points in Q2:', n_q2)
        print('Number of points in Q3:', n_q3)
        print('Number of points in Q4:', n_q4)
        print()

        #make scatter plot
        fig, ax = plt.subplots()
        ax.scatter(LH_wrf, SS_wrf*100, c=colors['all'], alpha=0.6, s=9)
        ax.set_xlabel('LH rate (K / s)')
        ax.set_ylabel('SS (%)')
        figtitle = 'Latent heating rate vs. supersaturation - ' + model_label
        fig.suptitle(figtitle)
        fig.set_size_inches(21, 12)

        #save figure
        outfile = FIG_DIR + versionstr + 'lh_vs_ss_all_' \
                    + model_label + '_figure.png'
        fig.savefig(outfile)
        plt.close()

if __name__ == "__main__":
    main()
