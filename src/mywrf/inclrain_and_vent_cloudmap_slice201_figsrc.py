"""
Same as inclrain_cloudmap_figsrc, but specifically for horizontal slice 42;
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
versionstr = 'v1_'

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

def main():
    """
    For both polluted and unpolluted model runs, create and save heatmap 
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][:, :, 201, :]
        meanfr = ncsecvars['meanfr'][:, :, 201, :]
        nconc = ncsecvars['nconc'][:, :, 201, :]
        ss_wrf = ncsecvars['ss_wrf'][:, :, 201, :]
        temp = ncsecvars['temp'][:, :, 201, :]
        w = ncsecvars['w'][:, :, 201, :]

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        ss_qss = w*A/(4*np.pi*D*nconc*meanfr)

        ncsecfile.close()

        lat = 1
        lon = 1
        
        for t in range(84):
            lwc_t = lwc[t, :, :]
            temp_t = temp[t, :, :]
            w_t = w[t, :, :]
            ss_qss_t = ss_qss[t, :, :]
            ss_wrf_t = ss_wrf[t, :, :]
            #make filter mask
            #mask = LWC_C > lwc_cutoff
            #mask = np.logical_and.reduce(( \
            #                            (LWC > lwc_cutoff), \
            #                            (nconc > 3.e6)))
            #mask = np.logical_and.reduce(( \
            #                            (LWC > lwc_cutoff), \
            #                            (np.abs(w) > 1), \
            #                            (np.abs(w) < 10)))
            #mask = np.logical_and.reduce(( \
            #                            (LWC_C > lwc_cutoff), \
            #                            (np.abs(w) > 5)))
            #mask = np.logical_and.reduce(( \
            #                            (lwc > lwc_cutoff), \
            #                            (temp > 273), \
            #                            (np.abs(w) > 4)))
            mask1 = np.logical_and.reduce(( \
                                        (lwc_t > lwc_cutoff), \
                                        (temp_t > 273), \
                                        (ss_wrf_t > 0), \
                                        (w_t > 2)))
            mask2 = np.logical_and.reduce(( \
                                        (lwc_t > lwc_cutoff), \
                                        (temp_t > 273), \
                                        (ss_wrf_t > -0.25), \
                                        (ss_qss_t < -0.25)))
            
            cloudmap = np.ones(np.shape(mask1))
            for i, row in enumerate(cloudmap):
                for j, col in enumerate(row):
                    if mask1[i, j]:
                        cloudmap[i, j] = 2
                    elif mask2[i, j]:
                        cloudmap[i, j] = 0

            #plot the supersaturations against each other with regression line
            fig, ax = plt.subplots()
            fig.set_size_inches(21, 12)

            im, _ = heatmap(cloudmap, lon, lat, ax=ax,
                            cmap=plt.get_cmap("coolwarm", 3))

            outfile = FIG_DIR + versionstr + \
                        'inclrain_and_vent_cloudmap_slice201_' \
                        + str(t) + '_' + model_label + '_figure.png'
            plt.savefig(outfile)
            plt.close(fig=fig)

def heatmap(data, x, y, ax=None, 
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    adapted from https://matplotlib.org/3.1.1/gallery/\
    images_contours_and_fields/image_annotated_heatmap.html
    """
    
    if not ax:
        ax = plt.gca()

    #plot heatmap
    im = ax.imshow(data, **kwargs)

    #create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    #hard-code 10 ticks per axis
    #x_int = (max(x) - min(x))/10.0
    #y_int = (max(y) - min(y))/10.0
    #ax.set_xticks(np.arange(min(x), max(x)+x_int, x_int))
    #ax.set_yticks(np.arange(min(y), max(y)+y_int, y_int))

    return im, cbar

if __name__ == "__main__":
    main()
