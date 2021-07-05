"""
Save w distribution for WCUs in HALO
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

change_CAS_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    make_and_save_w_data(w[filter_inds], change_CAS_corr, cutoff_bins, \
                                        full_ss, incl_rain, incl_vent)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_data(w, change_CAS_corr, cutoff_bins, \
                            full_ss, incl_rain, incl_vent):
    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(w, bins=30, density=False)
    w_data_dict = {'w': w, 'n': n, 'bins': bins}
    filename = 'w_hist_data.npy'
    np.save(DATA_DIR + filename, w_data_dict)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
