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
versionstr = 'v6_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

# bin diams
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    data_date_tuples = get_data_date_tuples(netcdf_files, good_dates)

    tot_nconc_alldates = None

    for data_date_tuple in data_date_tuples:
        smpsfile = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = data_date_tuple[2]

        tot_nconc = get_smps_nconc(smpsvars)
        tot_nconc_alldates = add_to_alldates_array(tot_nconc, \
                                            tot_nconc_alldates)
        print(tot_nconc_alldates)

    print(tot_nconc_alldates)
    make_and_save_tot_nconc_hist(tot_nconc_alldates, 'alldates')

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
        smps_filename = filename
        date = smps_filename[16:24]
        if date not in good_dates:
            continue 
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

def add_to_alldates_array(tot_nconc, tot_nconc_alldates):

    if tot_nconc_alldates is None:
        return tot_nconc
    else:
        return np.concatenate((tot_nconc_alldates, tot_nconc))

def make_and_save_tot_nconc_hist(tot_nconc, date):

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    n, b, p = ax.hist(tot_nconc, bins=3000, log=True, density=True)
    print(b)

    ax.set_xlabel('Aerosol number concentration, 11.1-469.8nm diameter (cm^-3)')
    ax.set_ylabel('Count')
    ax.set_title(date + ' tot nconc smps')
    ax.legend()
    outfile = FIG_DIR + versionstr + 'tot_nconc_hist_' \
            + date + '_figure.png'

    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
