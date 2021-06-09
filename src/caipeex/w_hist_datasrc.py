"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from caipeex import DATA_DIR, FIG_DIR
from caipeex.ss_functions import get_lwc_vs_t

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 1 

cutoff_bins = True 
incl_rain = False
incl_vent = False
full_ss = False

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    w_alldates = None

    for date in good_dates:
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cpd_dict = np.load(cpdfile, allow_pickle=True).item()

        lwc = get_lwc_vs_t(cpd_dict, cutoff_bins)
        temp = met_dict['data']['temp']
        w = met_dict['data']['w']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        w = w[filter_inds]

        w_alldates = add_to_alldates_array(w, w_alldates)

    print(w_alldates)
    make_and_save_w_data(w_alldates, 'alldates', \
            cutoff_bins, full_ss, incl_rain, incl_vent)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_data(w, label, cutoff_bins, full_ss, incl_rain, incl_vent):
    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(w, bins=30, density=False)
    w_data_dict = {'w': w, 'n': n, 'bins': bins}
    filename = 'w_hist_data_alldates.npy'
    np.save(DATA_DIR + filename, w_data_dict)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
