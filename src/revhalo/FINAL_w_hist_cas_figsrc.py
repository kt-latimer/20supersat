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
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[3]

lwc_filter_val = 1.e-5
w_cutoff = 2.5

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

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        w = adlr_dict['data']['w']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        w = w[filter_inds]

        w_alldates = add_to_alldates_array(w, w_alldates)

    print(w_alldates.shape)
    #make_and_save_w_hist(w_alldates, 'alldates', versionstr)

def add_to_alldates_array(w, w_alldates):

    if w_alldates is None:
        return w
    else:
        return np.concatenate((w_alldates, w))

def make_and_save_w_hist(w, label, versionstr):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(w, bins=30, color=magma_pink, density=False)
    ax.set_xlabel('w (m/s)')
    ax.set_ylabel('Data point count')
    outfile = FIG_DIR + versionstr + 'FINAL_w_hist_cas_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
