"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revhalo import DATA_DIR, FIG_DIR
from revhalo.ss_qss_calculations import get_lwc_from_cas
from revhalo.aero_calculations import get_kernel_weighted_nconc

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5 

change_cas_corr = False
cutoff_bins = False 
incl_rain = False
incl_vent = False

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    aero_nconc_alldates = None

    for date in good_dates:
        if date == '20140906' or date == '20140921': #dates dne for pcasp
            continue

        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        pcaspfile = DATA_DIR + 'npy_proc/PCASP_' + date + '.npy'
        pcasp_dict = np.load(pcaspfile, allow_pickle=True).item()

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        
        kernel = [1. for i in range(30)]
        aero_nconc = get_kernel_weighted_nconc(pcasp_dict, kernel)

        filter_inds = np.logical_and.reduce((
                        (lwc < lwc_filter_val), \
                        (aero_nconc < 5.e9), \
                        (temp > 273)))

        aero_nconc = aero_nconc[filter_inds]
        print(np.shape(aero_nconc))

        aero_nconc_alldates = \
            add_to_alldates_array(aero_nconc, aero_nconc_alldates)

        make_and_save_aero_nconc_hist(aero_nconc, date)

    make_and_save_aero_nconc_hist(aero_nconc_alldates, 'alldates')

def add_to_alldates_array(aero_nconc, aero_nconc_alldates):

    if aero_nconc_alldates is None:
        return aero_nconc
    else:
        return np.concatenate((aero_nconc_alldates, aero_nconc))

def make_and_save_aero_nconc_hist(aero_nconc, label):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(aero_nconc, bins=30, density=False)
    ax.set_xlabel('nconc (m^-3)')
    ax.set_ylabel('Count')
    ax.set_title(label + ' aerosol num conc distb' \
                    + ', change_cas_corr=' + str(change_cas_corr) \
                    + ', cutoff_bins=' + str(cutoff_bins))
    outfile = FIG_DIR + versionstr + 'aero_nconc_hist_cas_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
