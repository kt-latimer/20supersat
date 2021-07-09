"""
plot my vs file SMPS uap50al nconc to make sure everything is straight 
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np
import os

from goamazon import DATA_DIR, FIG_DIR, SMPS_bins

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'exp_fan_dates': (0.7, 0.7, 0.7), \
                'exp_halo_dates': colors_arr[2], \
                'wrf_poll': colors_arr[5], \
                'wrf_unpoll': colors_arr[8]}
# bin diams
SMPS_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    data_date_tuples = get_data_date_tuples(netcdf_files, good_dates)

    uap50_nconc_alldates = None

    for data_date_tuple in data_date_tuples:
        SMPS_file = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        SMPS_vars = SMPS_file.variables
        date = data_date_tuple[1]

        uap50_nconc = get_SMPS_nconc(SMPS_vars)
        uap50_nconc_alldates = add_to_alldates_array(uap50_nconc, \
                                            uap50_nconc_alldates)

    with open('fan_good_dates.txt', 'r') as readFile:
        fan_good_dates = [line.strip() for line in readFile.readlines()]

    fan_data_date_tuples = get_data_date_tuples(netcdf_files, fan_good_dates)

    uap50_fan_nconc_alldates = None

    for fan_data_date_tuple in fan_data_date_tuples:
        SMPS_file = Dataset(DATA_DIR + fan_data_date_tuple[0], 'r')
        SMPS_vars = SMPS_file.variables
        date = fan_data_date_tuple[1]

        uap50_fan_nconc = get_SMPS_nconc(SMPS_vars)
        uap50_fan_nconc_alldates = add_to_alldates_array(uap50_fan_nconc, \
                                            uap50_fan_nconc_alldates)

    make_and_save_uap50_nconc_hist(uap50_nconc_alldates, \
                    uap50_fan_nconc_alldates, 'alldates')

def get_data_date_tuples(netcdf_files, good_dates):
    """
    assumes files are listed in alpha order
    """

    data_date_tuples = []

    for filename in netcdf_files:
        if 'smps' not in filename:
            continue
        if 'cdf' not in filename:
            continue
        SMPS_filename = filename
        date = SMPS_filename[16:24]
        if date not in good_dates:
            continue 
        data_date_tuples.append((SMPS_filename, date))

    return data_date_tuples

def get_SMPS_nconc(SMPS_vars):

    SMPS_nconc = np.sum( \
            SMPS_dlogDp[0:42]*SMPS_vars['number_size_distribution'][...][:, 0:42], \
            axis=1)

    return SMPS_nconc

def add_to_alldates_array(uap50_nconc, uap50_nconc_alldates):

    if uap50_nconc_alldates is None:
        return uap50_nconc
    else:
        return np.concatenate((uap50_nconc_alldates, uap50_nconc))

def make_and_save_uap50_nconc_hist(uap50_nconc, uap50_fan_nconc, date):

    uap50_nconc_filter = uap50_nconc < 25000
    uap50_nconc = uap50_nconc[uap50_nconc_filter]

    uap50_fan_nconc_filter = uap50_fan_nconc < 25000
    uap50_fan_nconc = uap50_fan_nconc[uap50_fan_nconc_filter]

    fig, ax = plt.subplots()

    n, b, p = ax.hist(uap50_nconc, bins=50, density=True, \
        alpha=0.5, label='Dates used in this study', \
        color=colors_dict['exp_halo_dates'])
    ax.hist(uap50_fan_nconc, bins=b, density=True, alpha=0.5, \
                label='Dates used in Fan et al [1]', \
                color=colors_dict['exp_fan_dates'])
    ylim = ax.get_ylim()
    ax.plot([820, 820], ylim, linestyle='-', \
            color=colors_dict['wrf_poll'], \
            linewidth=3, label='WRF Polluted')
    ax.plot([0, 0], ylim, linestyle='-', \
            color=colors_dict['wrf_unpoll'], \
            linewidth=3, label='WRF Unpolluted')

    ax.set_xlabel(r'SMPS UAP$_{<50}$ number concentration (cm$^{-3}$)')
    ax.set_ylabel(r'$\frac{dn_{points}}{dN}$ (cm$^3$)')
    ax.legend()
    fig.suptitle('Aerosol concentration distributions - 11.1-50 nm')

    outfile = FIG_DIR + 'uap50_compare_nconc_hist_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
