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
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

# bin diams
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)
    data_date_tuples = get_data_date_tuples(netcdf_files)

    for data_date_tuple in data_date_tuples:
        smpsfile = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = data_date_tuple[2]

        my_smps_nconc = get_smps_nconc(smpsvars)
        file_smps_nconc = smpsvars['total_concentration'][...]

        make_and_save_nconc_scatter(my_smps_nconc, file_smps_nconc, date)

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
        smps_filename = filename
        date = smps_filename[16:24]
        for other_filename in netcdf_files:
            if 'uhsas' in other_filename \
            and date in other_filename \
            and 'cdf' in other_filename:
                uhsas_filename = other_filename
                break
        data_date_tuples.append((smps_filename, uhsas_filename, date))

    return data_date_tuples

def get_smps_nconc(smpsvars):

    smps_nconc = np.sum( \
            smps_dlogDp*smpsvars['number_size_distribution'][...], axis=1)

    return smps_nconc

def make_and_save_nconc_scatter(my_smps_nconc, file_smps_nconc, date):

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.scatter(my_smps_nconc, file_smps_nconc)

    ax.set_xlabel('my SMPS nconc (cm^-3)')
    ax.set_ylabel('file SMPS nconc (cm^-3)')
    ax.set_title(date + ' check total nconc')
    ax.legend()
    outfile = FIG_DIR + versionstr + 'check_smps_nconc_' \
            + date + '_figure.png'

    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
