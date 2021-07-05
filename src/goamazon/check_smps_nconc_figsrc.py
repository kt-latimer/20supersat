"""
plot my vs file SMPS total nconc to make sure everything is straight 
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

from goama import DATA_DIR, FIG_DIR, SMPS_bins

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

# bin diams
SMPS_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)
    data_date_tuples = get_data_date_tuples(netcdf_files)

    for data_date_tuple in data_date_tuples:
        SMPS_file = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        SMPS_vars = SMPS_file.variables
        date = data_date_tuple[2]

        my_SMPS_nconc = get_SMPS_nconc(SMPS_vars)
        file_SMPS_nconc = SMPS_vars['total_concentration'][...]

        make_and_save_nconc_scatter(my_SMPS_nconc, file_SMPS_nconc, date)

def get_data_date_tuples(netcdf_files):
    """
    assumes files are listed in alpha order
    """

    data_date_tuples = []

    for filename in netcdf_files:
        if 'smps' not in filename:
            break
        if 'cdf' not in filename:
            continue
        SMPS_filename = filename
        date = SMPS_filename[16:24]
        for other_filename in netcdf_files:
            if 'uhsas' in other_filename \
            and date in other_filename \
            and 'cdf' in other_filename:
                UHSAS_filename = other_filename
                break
        data_date_tuples.append((SMPS_filename, UHSAS_filename, date))

    return data_date_tuples

def get_SMPS_nconc(SMPS_vars):

    SMPS_nconc = np.sum( \
            SMPS_dlogDp*SMPS_vars['number_size_distribution'][...], axis=1)

    return SMPS_nconc

def make_and_save_nconc_scatter(my_SMPS_nconc, file_SMPS_nconc, date):

    fig, ax = plt.subplots()

    ax.scatter(my_SMPS_nconc, file_SMPS_nconc)

    ax.set_xlabel('my SMPS nconc (cm^-3)')
    ax.set_ylabel('file SMPS nconc (cm^-3)')
    ax.set_title(date + ' check total nconc')
    ax.legend()
    outfile = FIG_DIR + 'check_SMPS_nconc_' \
            + date + '_figure.png'

    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
