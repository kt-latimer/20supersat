"""
Plot rate of latent heat release from water condensation/evaporation versus
supersaturation WRF variables *for updrafts only* and print metadata (i.e. 
point count in each quadrant) to output file.
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

cdict = {'red':     [(0.0, 0, 248./255.),
                     (1.0, 186./255., 0)],
         'green':   [(0.0, 0, 236./255.),
                     (1.0, 63./255., 0)],
         'blue':    [(0.0, 0, 230./255.),
                     (1.0, 0./255., 0)]}
redfade = LinearSegmentedColormap('redfade', cdict)

versionstr = 'v4_'
lwc_cutoff = 1.e-5
w_cutoff = 1

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

def main():
    """
    for both C_BG and C_PI model configurations (see Fan 2018), plot LH rate vs
    SS, with marker color gradient corresponding to vertical wind velocity. 
    metadata output will be printed to file specified in slurm command.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncvars = ncfile.variables

        #get relevant variables from wrf output
        LH_wrf = ncvars['TEMPDIFFL'][...]
        LWC_wrf = ncvars['QCLOUD'][...]
        SS_wrf = ncvars['SSW'][...]
        W_wrf = ncvars['W'][...]
        #vertical wind velocity is on a staggered grid; take NN average to
        #reshape to mass grid
        W_wrf = (W_wrf[:,0:-1,:,:] + W_wrf[:,1:,:,:])/2

        #filter out downdrafts (i.e. enforce w > 0)
        #mask = W_wrf > 0
        mask = np.logical_and(W_wrf > w_cutoff, LWC_wrf > lwc_cutoff)

        #get metadata
        n_q1 = np.sum(np.logical_and.reduce(( \
                                    (LH_wrf > 0), \
                                    (SS_wrf > 0), \
                                    (W_wrf > w_cutoff), \
                                    (LWC_wrf > lwc_cutoff))))
        n_q2 = np.sum(np.logical_and.reduce(( \
                                    (LH_wrf < 0), \
                                    (SS_wrf > 0), \
                                    (W_wrf > w_cutoff), \
                                    (LWC_wrf > lwc_cutoff))))
        n_q3 = np.sum(np.logical_and.reduce(( \
                                    (LH_wrf < 0), \
                                    (SS_wrf < 0), \
                                    (W_wrf > w_cutoff), \
                                    (LWC_wrf > lwc_cutoff))))
        n_q4 = np.sum(np.logical_and.reduce(( \
                                    (LH_wrf > 0), \
                                    (SS_wrf < 0), \
                                    (W_wrf > w_cutoff), \
                                    (LWC_wrf > lwc_cutoff))))

        #print metadata to output file
        print(model_label)
        print('Number of points in Q1:', n_q1)
        print('Number of points in Q2:', n_q2)
        print('Number of points in Q3:', n_q3)
        print('Number of points in Q4:', n_q4)
        print()

        #make scatter plot
        fig, ax = plt.subplots()
        im = ax.scatter(LH_wrf[mask], SS_wrf[mask]*100, c=W_wrf[mask], \
                    s=9, cmap=redfade)
        ax.set_xlabel('LH rate (K / s)')
        ax.set_ylabel('SS (%)')
        figtitle = 'Latent heating rate vs. supersaturation - ' \
                    + model_label + '- w > ' + str(w_cutoff)
        fig.suptitle(figtitle)
        fig.set_size_inches(21, 12)
        fig.colorbar(im, ax=ax, label='w (m/s)')

        #save figure
        outfile = FIG_DIR + versionstr + 'lh_vs_ss_wpos_' \
                    + model_label + '_figure.png'
        fig.savefig(outfile)

        #clear up memory for next iteration of the for loop
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
