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
colors = {'bulk': '#BA3F00', 'edge': '#88720A'}

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
        PH = ncprimvars['PH'][...] #geopotential perturbation
        PHB = ncprimvars['PHB'][...] #geopotential base value
        U = ncprimvars['U'][...]
        V = ncprimvars['V'][...]

        #get altitude (geopotential) and realign to mass grid
        z = (PH + PHB)/g #altitude rel to sea level
        del PH, PHB
        z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2
        
        #align to mass grid
        u = (U[:, :, :, 0:-1] + U[:, :, :, 1:])/2.
        v = (V[:, :, 0:-1, :] + V[:, :, 1:, :])/2.

        #get magnitude of horizontal velocity
        uh = np.sqrt(u**2. + v**2.)

        del u, v, U, V #for memory

        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncprimfile.close()
        ncsecfile.close()

        #make filter mask
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

        #define all quantities w/ names and units for plotting
        datasets = [z, uh, lwc, meanr, nconc, temp, w, uh/w]
        abbrv_names = ['alt', 'horzwind', 'lwc', 'meanr', 'nconc', \
                        'temp', 'w', 'horzvertratio']
        long_names = ['Altitude', 'Horizonal wind speed', 'LWC', \
                        'Mean radius', 'Number concentration', 'Temperature', \
                        'Vertical wind velocity', 'Horz wind / vert wind']
        units = ['m', 'm/s', 'g/g', 'm', 'm^3', 'K', 'm/s', 'ratio']

        #loop through datasets
        for i, dataset in enumerate(datasets):

            #plot the supersaturations against each other with regression line
            fig, ax = plt.subplots()
            ax.hist(dataset[mask1], bins = 40, label='Cloud bulk', \
                    density=True, color=colors['bulk'], alpha=0.5)
            ax.hist(dataset[mask2], bins = 40, label='Cloud edge', \
                    density=True, color=colors['edge'], alpha=0.5)
            ax.set_xlabel(long_names[i] + ' (' + units[i] + ')')
            ax.set_ylabel('Distribution (' + units[i] + '^-1)')
            fig.set_size_inches(21, 12)

            #plot markers for min and max values of quantity in dataset for 
            #cloud bulk and cloud edge points
            bulk_max = np.max(dataset[mask1])
            bulk_min = np.min(dataset[mask1])
            edge_max = np.max(dataset[mask2])
            edge_min = np.min(dataset[mask2])

            ax.plot([bulk_max, bulk_max], ax.get_ylim(), linestyle='-.', \
                    color=colors['bulk'], label='Cloud bulk max')
            ax.plot([bulk_min, bulk_min], ax.get_ylim(), linestyle='--', \
                    color=colors['bulk'], label='Cloud bulk min')
            ax.plot([edge_max, edge_max], ax.get_ylim(), linestyle='-.', \
                    color=colors['edge'], label='Cloud edge max')
            ax.plot([edge_min, edge_min], ax.get_ylim(), linestyle='--', \
                    color=colors['edge'], label='Cloud edge min')

            fig.legend(loc=2)

            outfile = FIG_DIR + versionstr + 'inclrain_and_vent_hist_set_' \
                        + abbrv_names[i] + '_' + model_label + '_figure.png'
            plt.savefig(outfile)
            plt.close(fig=fig)

if __name__ == "__main__":
    main()
