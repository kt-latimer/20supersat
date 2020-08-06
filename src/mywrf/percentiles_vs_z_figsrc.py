"""
Same as inclrain_cloudmap_figsrc, but specifically for east/west slice 201;
create one figure for each time step in the model output 
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
versionstr = 'v2_'

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
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

lat = 1
lon = 1

def main():
    """
    For both polluted and unpolluted model runs, create and save heatmap 
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

        #get altitude (geopotential) and realign to mass grid
        z = (PH + PHB)/g #altitude rel to sea level
        del PH, PHB
        z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2

        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncprimfile.close()
        ncsecfile.close()
        
        mask = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2)))
        bulk_cutoff = np.percentile(lwc[mask], 5)
        
        by_alt_cutoffs = []
        by_alt_and_t_cutoffs_min = []
        by_alt_and_t_cutoffs_max = []
        by_alt_and_t_cutoffs_minus_sigma = []
        by_alt_and_t_cutoffs_plus_sigma = []
        z_coords = []

        for j in range(lwc.shape[1]):
            lwc_z = lwc[:, j, :, :]
            z_coords.append(np.mean(z[:, j, :, :]))
            mask_z = mask[:, j, :, :]
            print(np.sum(mask_z))
            if np.sum(mask_z) != 0:
                by_alt_cutoffs.append(np.percentile(lwc_z[mask[:, j, :, :]], 5))
                cutoffs_z = []
                for k in range(lwc.shape[0]):
                    lwc_z_t = lwc_z[k, :, :]
                    mask_z_t = mask[k, j, :, :]
                    if np.sum(mask_z_t) != 0:
                        cutoffs_z.append(np.percentile(lwc_z_t[mask_z_t], 5))
                    else:
                        cutoffs_z.append(np.nan)
                by_alt_and_t_cutoffs_min.append(np.nanmin(cutoffs_z))
                by_alt_and_t_cutoffs_max.append(np.nanmax(cutoffs_z))
                by_alt_and_t_cutoffs_minus_sigma.append(by_alt_cutoffs[-1] \
                                                - np.nanstd(cutoffs_z))
                by_alt_and_t_cutoffs_plus_sigma.append(by_alt_cutoffs[-1] \
                                                + np.nanstd(cutoffs_z))
            else:
                by_alt_cutoffs.append(np.nan)
                by_alt_and_t_cutoffs_min.append(np.nan)
                by_alt_and_t_cutoffs_max.append(np.nan)
                by_alt_and_t_cutoffs_minus_sigma.append(np.nan)
                by_alt_and_t_cutoffs_plus_sigma.append(np.nan)

        fig, ax = plt.subplots()
        fig.set_size_inches(6, 12)

        ax.plot(bulk_cutoff*np.ones(len(z_coords)), z_coords, color='black', \
                linewidth=3, marker="", label='LWC 5th perc. all times and \
                altitudes')
        #ax.fill_betweenx(z_coords, by_alt_and_t_cutoffs_min, \
        #                by_alt_and_t_cutoffs_max, color='red', alpha=0.5, \
        #                label='LWC 5th perc. vs altitude range across time')
        ax.fill_betweenx(z_coords, by_alt_and_t_cutoffs_minus_sigma, \
                        by_alt_and_t_cutoffs_plus_sigma, color='red', alpha=0.5, \
                        label='LWC 5th perc. vs altitude +/- 1 sigma across time')
        ax.plot(by_alt_cutoffs, z_coords, color='red', linewidth=2, \
                marker="", label='LWC 5th perc. all times vs altitude')
        ax.set_xlabel('LWC (kg/kg)')
        ax.set_ylabel('z (m)')
        ax.legend(loc=2)

        outfile = FIG_DIR + versionstr + 'percentiles_vs_z_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
