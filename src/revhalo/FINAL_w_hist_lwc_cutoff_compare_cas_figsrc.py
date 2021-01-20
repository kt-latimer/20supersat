"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from revhalo import DATA_DIR, FIG_DIR
from revhalo.ss_qss_calculations import get_lwc_from_cas

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[3]
magma_orange = colors_arr[7]

lwc_filter_val_1 = 1.e-5
lwc_filter_val_2 = 1.e-4
w_cutoff = 2

change_cas_corr = True
cutoff_bins = True 
incl_rain = True
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    w_alldates_1 = None
    w_alldates_2 = None

    for date in good_dates:
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        w = adlr_dict['data']['w']

        filter_inds_1 = np.logical_and.reduce((
                        (lwc > lwc_filter_val_1), \
                        (w > w_cutoff), \
                        (temp > 273)))
        filter_inds_2 = np.logical_and.reduce((
                        (lwc > lwc_filter_val_2), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        w_1 = w[filter_inds_1]
        w_2 = w[filter_inds_2]

        w_alldates_1 = add_to_alldates_array(w_1, w_alldates_1)
        w_alldates_2 = add_to_alldates_array(w_2, w_alldates_2)

    make_and_save_w_hist(w_alldates_1, w_alldates_2, 'alldates', versionstr)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_hist(w_1, w_2, label, versionstr):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    counts, bins, patches = ax.hist(w_1, bins=30, color=magma_pink, \
            alpha=0.5, density=False, label='LWC $>$ 1e-5')
    ax.hist(w_2, bins=bins, color=magma_orange, alpha=0.5, \
            density=False, label='LWC $>$ 1e-4')
    #counts, bins, patches = ax.hist(w_1, bins=30, linewidth=4, \
    #        facecolor=(0, 0, 0, 0.0), edgecolor=magma_pink, \
    #        histtype='stepfilled', density=False, label='LWC $>$ 1e-5')
    #ax.hist(w_2, bins=30, linewidth=4, facecolor=(0, 0, 0, 0.0), \
    #        edgecolor=magma_orange, histtype='stepfilled', density=False, \
    #        label='LWC $>$ 1e-4')
    #counts, bins, patches = ax.hist(w_1, bins=30, \
    #        color=magma_pink, alpha=0.5, \
    #        density=False, label='LWC $>$ 1e-5')
    #ax.hist(w_1, bins=bins, linewidth=4, \
    #        facecolor=(0, 0, 0, 0.0), edgecolor=magma_pink, \
    #        histtype='stepfilled', density=False)
    #ax.hist(w_2, bins=30, alpha=0.5,\
    #        color=magma_orange, density=False, \
    #        label='LWC $>$ 1e-4', linewidth=0)
    ax.set_xlabel('w (m/s)')
    ax.set_ylabel('Data point count')
    ax.legend()

    outfile = FIG_DIR + versionstr + 'FINAL_w_hist_lwc_cutoff_compare_cas_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
