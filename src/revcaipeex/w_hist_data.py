"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_lwc

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

#change_cas_corr = False
#cutoff_bins = False 
#incl_rain = False
#incl_vent = False
#full_ss = False

def main():
    
    if len(sys.argv) > 1:
        versionnum = int(sys.argv[1])
        (dummy_bool, cutoff_bins, full_ss, \
            incl_rain, incl_vent) = get_boolean_params(versionnum)
        versionstr = 'v' + str(versionnum) + '_'

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    w_alldates = None

    for date in good_dates:
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cpd_dict = np.load(cpdfile, allow_pickle=True).item()

        lwc = get_lwc(cpd_dict,cutoff_bins)
        temp = met_dict['data']['temp']
        w = met_dict['data']['w']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        w = w[filter_inds]

        w_alldates = add_to_alldates_array(w, w_alldates)

        #make_and_save_w_data(w, date, versionstr, change_cas_corr, \
        #                            cutoff_bins, full_ss, incl_rain, incl_vent)

    print(w_alldates)
    make_and_save_w_data(w_alldates, 'alldates', versionstr, \
            cutoff_bins, full_ss, incl_rain, incl_vent)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_data(w, label, versionstr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    n, bins, patches = ax.hist(w, bins=30, density=False)
    w_data_dict = {'w': w, 'n': n, 'bins': bins}
    filename = versionstr + 'w_hist_cas_data_alldates.npy'
    np.save(DATA_DIR + filename, w_data_dict)
    plt.close(fig=fig)    

def get_boolean_params(versionnum):

    versionnum = versionnum - 1 #for modular arithmetic
    
    if versionnum > 23:
        return versionnum

    if versionnum < 12:
        change_cas_corr = False
    else:
        change_cas_corr = True

    if versionnum % 12 < 6:
        cutoff_bins = False
    else:
        cutoff_bins = True

    if versionnum % 6 < 3:
        full_ss = False
    else:
        full_ss = True

    if versionnum % 3 == 0:
        incl_rain = False
    else:
        incl_rain = True

    if versionnum % 3 == 2:
        incl_vent = True
    else:
        incl_vent = False

    return (change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

if __name__ == "__main__":
    main()
