"""
plot my vs file SMPS total nconc to make sure everything is straight 
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np
import os

from goama import DATA_DIR, FIG_DIR, SMPS_bins

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

    tot_nconc_alldates = None

    for data_date_tuple in data_date_tuples:
        SMPS_file = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        SMPS_vars = SMPS_file.variables
        date = data_date_tuple[1]

        tot_nconc = get_SMPS_nconc(SMPS_vars)
        tot_nconc_alldates = add_to_alldates_array(tot_nconc, \
                                            tot_nconc_alldates)

    with open('fan_good_dates.txt', 'r') as readFile:
        fan_good_dates = [line.strip() for line in readFile.readlines()]

    fan_data_date_tuples = get_data_date_tuples(netcdf_files, fan_good_dates)
    tot_fan_nconc_alldates = None

    for fan_data_date_tuple in fan_data_date_tuples:
        SMPS_file = Dataset(DATA_DIR + fan_data_date_tuple[0], 'r')
        SMPS_vars = SMPS_file.variables
        date = fan_data_date_tuple[1]

        tot_fan_nconc = get_SMPS_nconc(SMPS_vars)
        tot_fan_nconc_alldates = add_to_alldates_array(tot_fan_nconc, \
                                            tot_fan_nconc_alldates)

    make_and_save_tot_nconc_hist(tot_nconc_alldates, tot_fan_nconc_alldates)

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
            SMPS_dlogDp*SMPS_vars['number_size_distribution'][...], axis=1)

    return SMPS_nconc

def add_to_alldates_array(tot_nconc, tot_nconc_alldates):

    if tot_nconc_alldates is None:
        return tot_nconc
    else:
        return np.concatenate((tot_nconc_alldates, tot_nconc))

def make_and_save_tot_nconc_hist(tot_nconc, tot_fan_nconc):

    tot_nconc_filter = tot_nconc < 25000
    tot_nconc = tot_nconc[tot_nconc_filter]

    tot_fan_nconc_filter = tot_fan_nconc < 25000
    tot_fan_nconc = tot_fan_nconc[tot_fan_nconc_filter]

    fig, ax = plt.subplots()

    n, b, p = ax.hist(tot_nconc, bins=50, density=True, \
        alpha=0.5, label='Dates used in this study', \
        color=colors_dict['exp_halo_dates'])
    ax.hist(tot_fan_nconc, bins=b, density=True, alpha=0.7, \
                label='Dates used in Fan et al [1]', \
                color=colors_dict['exp_fan_dates'])
    ylim = ax.get_ylim()
    ax.plot([950, 950], ylim, linestyle='-', \
            color=colors_dict['wrf_poll'], \
            linewidth=3, label='WRF Polluted')
    ax.plot([130, 130], ylim, linestyle='-', \
            color=colors_dict['wrf_unpoll'], \
            linewidth=3, label='WRF Unpolluted')

    ax.set_xlabel(r'Total SMPS number concentration (cm$^{-3}$)')
    ax.set_ylabel(r'$\frac{dn_{points}}{dN}$ (cm$^3$)')
    ax.legend()
    fig.suptitle('Aerosol concentration distributions - 11.1-469.8 nm')

    outfile = FIG_DIR + 'FINAL_tot_compare_nconc_hist_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
