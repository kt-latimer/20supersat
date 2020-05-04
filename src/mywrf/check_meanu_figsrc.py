"""
double-check my calculated terminal (number- and mass-weighted) fall speeds \
for rain and cloud drops 
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
lwc_cutoff = 1.e-5

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

def main():
    """
    For both polluted and unpolluted model runs, plot my u_term vs wrf's
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir +
                    'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get WRF mass- and number-weighted fall velocities for cloud and rain 
        MEANUCLOUD = ncprimvars['FNC'][...]
        MEANURAIN = ncprimvars['FNR'][...]
        MEANUR3CLOUD = ncprimvars['FC'][...]
        MEANUR3RAIN = ncprimvars['FR'][...]

        #get calc'd mass- and number-weighted fall velocities for cloud and rain 
        meanucloud = ncsecvars['meanucloud'][...]
        meanurain = ncsecvars['meanurain'][...]
        meanur3cloud = ncsecvars['meanur3cloud'][...]
        meanur3rain = ncsecvars['meanur3rain'][...]

        #get lwc
        lwc = ncsecvars['lwc_cloud'][...]

        ncprimfile.close()
        ncsecfile.close()

        #make mask to filter out low LWC points
        mask = lwc > lwc_cutoff

        #apply filter
        MEANUCLOUD = MEANUCLOUD[mask]
        MEANURAIN = MEANURAIN[mask]
        MEANUR3CLOUD = MEANUR3CLOUD[mask]
        MEANUR3RAIN = MEANUR3RAIN[mask]
        meanucloud = meanucloud[mask]
        meanurain = meanurain[mask]
        meanur3cloud = meanur3cloud[mask]
        meanur3rain = meanur3rain[mask]

        #linear regression params
        mucloud, bucloud, Rucloud, sigucloud = \
                        linregress(meanucloud, MEANUCLOUD)
        murain, burain, Rurain, sigurain = \
                        linregress(meanurain, MEANURAIN)
        mur3cloud, bur3cloud, Rur3cloud, sigur3cloud = \
                        linregress(meanur3cloud, MEANUR3CLOUD)
        mur3rain, bur3rain, Rur3rain, sigur3rain = \
                        linregress(meanur3rain, MEANUR3RAIN)

        #get limits of the data for plotting purposes
        xlim_max_00 = np.max(np.array( \
                        [np.max(meanucloud), \
                         np.max(MEANUCLOUD)]))
        xlim_min_00 = np.min(np.array( \
                        [np.min(meanucloud), \
                         np.min(MEANUCLOUD)]))
        ax_lims_00 = np.array([xlim_min_00, xlim_max_00])

        xlim_max_01 = np.max(np.array( \
                        [np.max(meanurain), \
                         np.max(MEANURAIN)]))
        xlim_min_01 = np.min(np.array( \
                        [np.min(meanurain), \
                         np.min(MEANURAIN)]))
        ax_lims_01 = np.array([xlim_min_01, xlim_max_01])

        xlim_max_10 = np.max(np.array( \
                        [np.max(meanur3cloud), \
                         np.max(MEANUR3CLOUD)]))
        xlim_min_10 = np.min(np.array( \
                        [np.min(meanur3cloud), \
                         np.min(MEANUR3CLOUD)]))
        ax_lims_10 = np.array([xlim_min_10, xlim_max_10])

        xlim_max_11 = np.max(np.array( \
                        [np.max(meanur3rain), \
                         np.max(MEANUR3RAIN)]))
        xlim_min_11 = np.min(np.array( \
                        [np.min(meanur3rain), \
                         np.min(MEANUR3RAIN)]))
        ax_lims_11 = np.array([xlim_min_11, xlim_max_11])

        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots(2,2)
        ax[0][0].scatter(meanucloud, MEANUCLOUD, c=colors['ss'])
        ax[0][0].plot(ax_lims_00, np.add(bucloud, mucloud*ax_lims_00), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(mucloud, decimals=2)) + \
                                ', R^2 = ' + str(np.round(Rucloud**2, decimals=2))))
        ax[0][0].set_title('Number-weighted fall speed, cloud')
        ax[0][0].set_xlabel('Calculated')
        ax[0][0].set_ylabel('WRF output')
        ax[0][0].legend(loc=2)

        ax[0][1].scatter(meanurain, MEANURAIN, c=colors['ss'])
        ax[0][1].plot(ax_lims_01, np.add(burain, murain*ax_lims_01), \
                       c=colors['line'], \
                       linestyle='dashed', \
                       linewidth=3, \
                       label=('m = ' + str(np.round(murain, decimals=2)) + \
                               ', R^2 = ' + str(np.round(Rurain**2, decimals=2))))
        ax[0][1].set_title('Number-weighted fall speed, rain')
        ax[0][1].set_xlabel('Calculated')
        ax[0][1].set_ylabel('WRF output')
        ax[0][1].legend(loc=2)

        ax[1][0].scatter(meanur3cloud, MEANUR3CLOUD, c=colors['ss'])
        ax[1][0].plot(ax_lims_10, np.add(bur3cloud, mur3cloud*ax_lims_10), \
                       c=colors['line'], \
                       linestyle='dashed', \
                       linewidth=3, \
                       label=('m = ' + str(np.round(mur3cloud, decimals=2)) + \
                               ', R^2 = ' + str(np.round(Rur3cloud**2, decimals=2))))
        ax[1][0].set_title('Mass-weighted fall speed, cloud')
        ax[1][0].set_xlabel('Calculated')
        ax[1][0].set_ylabel('WRF output')
        ax[1][0].legend(loc=2)

        ax[1][1].scatter(meanur3rain, MEANUR3RAIN, c=colors['ss'])
        ax[1][1].plot(ax_lims_11, np.add(bur3rain, mur3rain*ax_lims_11), \
                      c=colors['line'], \
                      linestyle='dashed', \
                      linewidth=3, \
                      label=('m = ' + str(np.round(mur3rain, decimals=2)) + \
                              ', R^2 = ' + str(np.round(Rur3rain**2, decimals=2))))
        ax[1][1].set_title('Mass-weighted fall speed, rain')
        ax[1][1].set_xlabel('Calculated')
        ax[1][1].set_ylabel('WRF output')
        ax[1][1].legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'check_meanu_' \
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
