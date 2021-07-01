"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_vs_t, get_full_spectrum_dict

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 1

rmax = 102.

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    lwc_alldates = None

    for date in good_dates:
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        full_spectrum_dict = get_full_spectrum_dict(cas_dict, \
                                    cip_dict, change_cas_corr)

        lwc = get_lwc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, rmax)
        temp = adlr_dict['data']['temp']
        w = adlr_dict['data']['w']
        z = adlr_dict['data']['alt']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (z > 1500), \
                        (z < 2500), \
                        (temp > 273)))

        lwc = lwc[filter_inds]

        lwc_alldates = add_to_alldates_array(lwc, lwc_alldates)

    print(np.mean(lwc_alldates))
    print(np.nanmean(lwc_alldates))

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

if __name__ == "__main__":
    main()
