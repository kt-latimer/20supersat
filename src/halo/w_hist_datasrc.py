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

rmax = 102.e-6

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    w_alldates = None

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

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        w = w[filter_inds]

        w_alldates = add_to_alldates_array(w, w_alldates)

    print(w_alldates)
    make_and_save_w_data(w_alldates, 'alldates', change_cas_corr, \
                        cutoff_bins, full_ss, incl_rain, incl_vent)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_data(w, label, change_cas_corr, cutoff_bins, \
                                    full_ss, incl_rain, incl_vent):
    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(w, bins=30, density=False)
    w_data_dict = {'w': w, 'n': n, 'bins': bins}
    filename = 'w_hist_cas_data_alldates.npy'
    np.save(DATA_DIR + filename, w_data_dict)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
