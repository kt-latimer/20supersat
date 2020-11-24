"""
plot range of SMPS spectra by day wrt concentration of UAF_50 ptcls
(as defined in Fan 2018)
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
smps_diams = np.sqrt(SMPS_bins['upper']*SMPS_bins['lower'])
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    netcdf_files = os.listdir(DATA_DIR)
    data_date_tuples = get_data_date_tuples(netcdf_files)

    for data_date_tuple in data_date_tuples:
        smpsfile = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = data_date_tuple[2]

        uaf50_nconc = get_uaf50_nconc(smpsvars)

        pcen0 = np.percentile(uaf50_nconc, 0, interpolation='nearest')
        pcen25 = np.percentile(uaf50_nconc, 25, interpolation='nearest')
        pcen50 = np.percentile(uaf50_nconc, 50, interpolation='nearest')
        pcen75 = np.percentile(uaf50_nconc, 75, interpolation='nearest')
        pcen100 = np.percentile(uaf50_nconc, 100, interpolation='nearest')

        pcen0_ind = np.where(uaf50_nconc == pcen0)[0][0]
        pcen25_ind = np.where(uaf50_nconc == pcen25)[0][0]
        pcen50_ind = np.where(uaf50_nconc == pcen50)[0][0]
        pcen75_ind = np.where(uaf50_nconc == pcen75)[0][0]
        pcen100_ind = np.where(uaf50_nconc == pcen100)[0][0]

        make_and_save_range_of_spectra_plot(pcen0, pcen0_ind, \
                                            pcen25, pcen25_ind, \
                                            pcen50, pcen50_ind, \
                                            pcen75, pcen75_ind, \
                                            pcen100, pcen100_ind, \
                                            smpsvars, date)
        print(date)
        print(pcen0, pcen25, pcen50, pcen75, pcen100)

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

def get_uaf50_nconc(smpsvars):

    uaf50_nconc = np.sum( \
            smps_dlogDp[0:42]*smpsvars['number_size_distribution'][...][:, 0:42], \
            axis=1)

    return uaf50_nconc

def make_and_save_range_of_spectra_plot(pcen0, pcen0_ind, \
                                        pcen25, pcen25_ind, \
                                        pcen50, pcen50_ind, \
                                        pcen75, pcen75_ind, \
                                        pcen100, pcen100_ind, \
                                        smpsvars, date):

    smps_spectra = smpsvars['number_size_distribution'][...]

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.fill_between(smps_diams*1.e9, \
                    smps_spectra[pcen0_ind], \
                    smps_spectra[pcen100_ind], \
                    color='b', \
                    alpha=0.5, \
                    label='Range')
    ax.plot(smps_diams*1.e9, \
            smps_spectra[pcen25_ind], \
            color='b', \
            linestyle='--', \
            label='Quartiles')
    ax.plot(smps_diams*1.e9, \
            smps_spectra[pcen75_ind], \
            color='b', \
            linestyle='--', \
            label='_nolabel_')
    ax.plot(smps_diams*1.e9, \
            smps_spectra[pcen50_ind], \
            color='b', \
            label='Median')

    ax.set_xlabel('Diameter (nm)')
    ax.set_ylabel('dN/dlogDp (cm^-3)')
    ax.set_title(date + ' range of aerosol size distributions')
    ax.legend()
    outfile = FIG_DIR + versionstr + 'ranges_of_spectra_' \
            + date + '_figure.png'

    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
