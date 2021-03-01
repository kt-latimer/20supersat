"""
plot my vs file SMPS uap50al nconc to make sure everything is straight 
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

from goama import DATA_DIR, FIG_DIR, SMPS_bins

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

# bin diams
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    data_date_tuples = get_data_date_tuples(netcdf_files, good_dates)

    uap50_nconc_alldates = None

    for data_date_tuple in data_date_tuples:
        smpsfile = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = data_date_tuple[1]

        uap50_nconc = get_smps_nconc(smpsvars)
        print(date, np.mean(uap50_nconc))
        uap50_nconc_alldates = add_to_alldates_array(uap50_nconc, \
                                            uap50_nconc_alldates)

    with open('fan_good_dates.txt', 'r') as readFile:
        fan_good_dates = [line.strip() for line in readFile.readlines()]

    fan_data_date_tuples = get_data_date_tuples(netcdf_files, fan_good_dates)

    uap50_fan_nconc_alldates = None

    for fan_data_date_tuple in fan_data_date_tuples:
        smpsfile = Dataset(DATA_DIR + fan_data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = fan_data_date_tuple[1]

        uap50_fan_nconc = get_smps_nconc(smpsvars)
        print(date, np.mean(uap50_fan_nconc))
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
        smps_filename = filename
        date = smps_filename[16:24]
        if date not in good_dates:
            continue 
        data_date_tuples.append((smps_filename, date))

    return data_date_tuples

def get_smps_nconc(smpsvars):

    smps_nconc = np.sum( \
            smps_dlogDp[0:42]*smpsvars['number_size_distribution'][...][:, 0:42], \
            axis=1)

    return smps_nconc

def add_to_alldates_array(uap50_nconc, uap50_nconc_alldates):

    if uap50_nconc_alldates is None:
        return uap50_nconc
    else:
        return np.concatenate((uap50_nconc_alldates, uap50_nconc))

def make_and_save_uap50_nconc_hist(uap50_nconc, uap50_fan_nconc, date):

    uap50_nconc_filter = uap50_nconc < 25000
    print('before filter: ', np.shape(uap50_nconc))
    uap50_nconc = uap50_nconc[uap50_nconc_filter]
    print('after filter: ', np.shape(uap50_nconc))

    uap50_fan_nconc_filter = uap50_fan_nconc < 25000
    print('before filter: ', np.shape(uap50_fan_nconc))
    uap50_fan_nconc = uap50_fan_nconc[uap50_fan_nconc_filter]
    print('after filter: ', np.shape(uap50_fan_nconc))

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)

    n, b, p = ax.hist(uap50_nconc, bins=50, density=True, \
        alpha=0.5, label='HALO flight dates', color='r')
    ax.hist(uap50_fan_nconc, bins=b, density=True, alpha=0.5, \
                label='Dates in Fan et al study', color='b')
    ylim = ax.get_ylim()
    ax.plot([130, 130], ylim, linestyle='--', \
            color='b', linewidth=4, \
            label='Fan et al model, unpolluted')
    ax.plot([820, 820], ylim, linestyle=':', \
            color='b', linewidth=4, \
            label='Fan et al model, polluted')

    ax.set_xlabel('Number concentration (cm^-3)')
    ax.set_ylabel('Count')
    ax.set_title(date + ' uap50 nconc smps')
    fig.legend()

    outfile = FIG_DIR + versionstr + 'uap50_compare_nconc_hist_' \
            + date + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
