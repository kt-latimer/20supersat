"""
Plot vertical wind velocity distributions for both simulation and 
experimental datasets
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from wrf import DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'halo': colors_arr[2], 'wrf_poll': colors_arr[5], \
                                    'wrf_unpoll': colors_arr[8]}

HALO_DATA_DIR = '/global/home/users/kalatimer/proj/20supersat/data/halo/'
CAIPEEX_DATA_DIR = \
    '/global/home/users/kalatimer/proj/20supersat/data/caipeex/'

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():

    ### halo ###
    halo_dict = np.load(HALO_DATA_DIR + 'w_hist_data.npy', \
                        allow_pickle=True).item()
    w_halo = halo_dict['w']

    ### wrf ###
    wrf_dict = np.load(DATA_DIR + 'filtered_data_dict.npy', \
                        allow_pickle=True).item()

    w_dict = {'Polluted': None, 'Unpolluted': None}
    
    for case_label in case_label_dict.keys():
        case_filtered_data_dict = wrf_dict[case_label]
        w = case_filtered_data_dict['w']
        w_dict[case_label] = {'w': w}
    
    make_and_save_w_hist(w_dict['Polluted']['w'], \
                          w_dict['Unpolluted']['w'], \
                          w_halo, w_caipeex)

def make_and_save_w_hist(w_wrf_polluted, w_wrf_unpolluted, \
                        w_halo, w_caipeex):

    fig, ax = plt.subplots()
    n_wrf, bins_wrf, patches_wrf = ax.hist(w_wrf_polluted, bins=30, \
            density=True, label='WRF Polluted', facecolor=(0, 0, 0, 0.0), \
            edgecolor=colors_dict['wrf_poll'], \
            histtype='stepfilled', linewidth=3)
    ax.hist(w_wrf_unpolluted, bins=bins_wrf, density=True, \
            label='WRF Unpolluted', facecolor=(0, 0, 0, 0.0), \
            edgecolor=colors_dict['wrf_unpoll'], \
            histtype='stepfilled', linewidth=3) 
    ax.hist(w_halo, bins=bins_wrf, density=True, label='HALO', \
            facecolor=(0, 0, 0, 0.0), edgecolor=colors_dict['halo'], \
            histtype='stepfilled', linewidth=3)
    ax.set_xlabel('w (m/s)')
    ax.set_ylabel(r'$\frac{dn_{points}}{dw}$ (s/m)')
    fig.suptitle('Vertical wind velocity distributions')
    plt.legend()

    outfile = FIG_DIR + 'combined_w_hist_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
