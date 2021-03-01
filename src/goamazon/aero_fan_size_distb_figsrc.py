"""
plot daily average SMPS and UHSAS spectra to make sure they align
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

from goama import DATA_DIR, FIG_DIR, SMPS_bins, UHSAS_bins

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', \
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
fan_colors = {'polluted': '#87cfeb', 'unpolluted': '#0b0bf7'}

# bin diams
smps_diams = np.sqrt(SMPS_bins['upper']*SMPS_bins['lower'])
uhsas_diams = np.sqrt(UHSAS_bins['upper']*UHSAS_bins['lower'])
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])
uhsas_dlogDp = np.log10(UHSAS_bins['upper']/UHSAS_bins['lower'])

n_spectra = 10
unpoll_cutoff = 18 

def main():

    fan_aero_size_distb = get_fan_aero_size_distb()
    
    netcdf_files = os.listdir(DATA_DIR)

    with open('fan_good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    data_date_tuples = get_data_date_tuples(netcdf_files, good_dates)

    for data_date_tuple in data_date_tuples:
        smpsfile = Dataset(DATA_DIR + data_date_tuple[0], 'r')
        smpsvars = smpsfile.variables
        date = data_date_tuple[1]

        (smps_spectra, smps_times, uap50_nconcs) = \
                get_smps_spectra_and_times_and_uap50_nconcs(smpsvars)

        #smps_labels = ['SMPS ' + str(t) for t in smps_times]
        smps_labels = ['UAP_50 conc = ' + str(np.round(n)) + ' cm^-3' \
                        for n in uap50_nconcs]

        make_and_save_aero_size_distb_plot(smps_spectra, smps_labels, \
                                            fan_aero_size_distb, date)

def get_fan_aero_size_distb():

    #get aerosol psd extracted from fig S5C of fan et al 2018
    fan_aero_size_distb = []
    with open(DATA_DIR + 'fan_aero_size_distb.csv', 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            fan_aero_size_distb.append(row)
    fan_aero_size_distb = np.array(fan_aero_size_distb)
    fan_aero_size_distb = \
        fan_aero_size_distb[fan_aero_size_distb[:,0].argsort()]

    return fan_aero_size_distb

def match_two_arrays(arr1, arr2):

    """
    Return: (inds1, inds2) where arr1[inds1] = arr2[inds2].
    Assumes arr1 and arr2 are both sorted in the same order (ie time series)
    """
    inds1 = []
    inds2 = []
    startind2 = 0
    for i1, x1 in enumerate(arr1):
        for i2, x2 in enumerate(arr2[startind2:]):
            if x1 == x2:
                inds1.append(i1)
                inds2.append(i2+startind2)
                startind2 = i2 + startind2 + 1
                break
    return(inds1, inds2)

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

def get_smps_spectra_and_times_and_uap50_nconcs(variables):

    size_distb = variables['number_size_distribution'][...]
    time = variables['time'][...]

    n_steps_per_sample = int(np.ceil(np.shape(size_distb)[0]/n_spectra))

    averaged_spectra = []
    uap50_nconcs = []
    spectrum_dates = []

    for i in range(n_spectra - 1):
        averaged_spectra.append(np.mean( \
            size_distb[i*n_steps_per_sample:(i+1)*n_steps_per_sample], axis=0))
        uap50_nconcs.append(get_uap50_nconc(variables, \
                            i*n_steps_per_sample, \
                            (i+1)*n_steps_per_sample))
        spectrum_dates.append(time[i*n_steps_per_sample])

    averaged_spectra.append(np.mean( \
        size_distb[(n_spectra-1)*n_steps_per_sample:], axis=0))
    uap50_nconcs.append(get_uap50_nconc(variables, \
                        (n_spectra-1)*n_steps_per_sample, \
                        None))
    spectrum_dates.append(time[(n_spectra-1)*n_steps_per_sample])

    return(averaged_spectra, spectrum_dates, uap50_nconcs)

def get_uap50_nconc(smpsvars, start_t_ind, end_t_ind):

    dNlogDp = smpsvars['number_size_distribution'][...]

    if end_t_ind == None:
        uap50_nconc = np.mean(np.sum( \
                smps_dlogDp[0:42]*dNlogDp[start_t_ind:, 0:42], \
                axis=1), \
                axis=0)
    else:
        uap50_nconc = np.mean(np.sum( \
                smps_dlogDp[0:42]*dNlogDp[start_t_ind:end_t_ind, 0:42], \
                axis=1), \
                axis=0)

    return uap50_nconc

def get_uhsas_spectra_and_times(variables):

    time = variables['time'][...]
    nconc = variables['size_distribution'][...]
    samp_vol = np.array([variables['sampling_volume'][...] for i in range(99)])
    dlogDp = np.array([uhsas_dlogDp for i in range(np.shape(time)[0])])
    size_distb = nconc/samp_vol.T/dlogDp

    n_steps_per_sample = int(np.ceil(np.shape(size_distb)[0]/n_spectra))

    averaged_spectra = []
    spectrum_dates = []

    for i in range(n_spectra - 1):
        averaged_spectra.append(np.mean( \
            size_distb[i*n_steps_per_sample:(i+1)*n_steps_per_sample], axis=0))
        spectrum_dates.append(time[i*n_steps_per_sample])

    averaged_spectra.append(np.mean( \
        size_distb[(n_spectra-1)*n_steps_per_sample:], axis=0))
    spectrum_dates.append(time[(n_spectra-1)*n_steps_per_sample])

    return(averaged_spectra, spectrum_dates)

def make_and_save_aero_size_distb_plot(smps_spectra, smps_labels, \
                                        fan_aero_size_distb, date):

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    for i in range(n_spectra):
        ax.plot(smps_diams*1.e9, \
                smps_spectra[i], \
                color=colors[i], \
                label=smps_labels[i])
        #ax.plot(uhsas_diams*1.e9, \
        #        uhsas_spectra[i], \
        #        color=colors[i], \
        #        label=uhsas_labels[i], \
        #        linestyle='--')
    ax.plot(np.power(10, fan_aero_size_distb[:, 0]), \
            fan_aero_size_distb[:, 1], \
            label='Fan WRF simulation - polluted case', \
            color=fan_colors['polluted'], \
            linewidth=6)
    ax.plot(np.power(10, fan_aero_size_distb[unpoll_cutoff:, 0]), \
            fan_aero_size_distb[unpoll_cutoff:, 1], \
            label='Fan WRF simulation - unpolluted case', \
            color=fan_colors['unpolluted'], \
            linewidth=6, \
            linestyle='--')
        
    ax.set_xlabel('Diameter (nm)')
    ax.set_xscale('log')
    ax.set_ylabel('dN/dlogDp (cm^-3)')
    ax.set_ylim(bottom=0)
    ax.set_title(date + ' aerosol size distributions')
    ax.legend()
    outfile = FIG_DIR + versionstr + 'aero_fan_size_distb_' \
            + date + '_figure.png'

    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
