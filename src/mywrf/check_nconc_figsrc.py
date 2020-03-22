"""
double-check my calculated cloud droplet number concentration
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
versionstr = 'v4_'
lwc_cutoff = 1.e-5

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

def main():
    """
    For both polluted and unpolluted model runs, plot my nconc versus wrf's
    nconc
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir + 'wrfout_d01_secondary_vars', 'r')
        ncsecvars = ncsecfile.variables
        
        #get WRF cloud number conc (kg^-1)
        LWC = ncprimvars['QCLOUD'][...]
        NCONC = ncprimvars['QNCLOUD'][...]

        #get my cloud number conc (m^-3) and air density (kg/m^-3)
        nconc = ncsecvars['nconc'][...]
        rho_air = ncsecvars['rho_air'][...]

        #convert WRF conc to volumetric
        NCONC = NCONC*rho_air

        #make mask to filter out low LWC points
        mask = LWC > lwc_cutoff

        #apply filter
        nconc = nconc[mask]
        NCONC = NCONC[mask]

        ##flatten both number concentration arrays to 1D
        #nconc = nconc.flatten()
        #NCONC = NCONC.flatten()

        m, b, R, sig = linregress(nconc, NCONC)
        print(m, b, R**2)

        #get limits of the data for plotting purposes
        xlim_max = np.max(np.array( \
                        [np.max(nconc), \
                         np.max(NCONC)]))
        xlim_min = np.min(np.array( \
                        [np.min(nconc), \
                         np.min(NCONC)]))
        ax_lims = np.array([xlim_min, xlim_max])
        print(ax_lims)

        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(nconc, NCONC, c=colors['ss'])
        ax.plot(ax_lims, np.add(b, m*ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_xlabel('my nconc')
        ax.set_ylabel('WRF nconc')
        fig.legend()
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'check_nconc_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)

def ts(message):
    """
    print time stamp along with message
    """
    print(message, datetime.now())
    return

if __name__ == "__main__":
    main()
