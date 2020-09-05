"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from revhalo import DATA_DIR, FIG_DIR
from revhalo.ss_qss_calculations import get_meanr_vs_t_from_cas, get_lwc_from_cas

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5 
w_cutoff = -100

change_cas_corr = False
cutoff_bins = False 
incl_rain = False
incl_vent = False
full_ss = False

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    meanr_alldates = None

    for date in good_dates:
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        print(np.max(lwc))
        temp = adlr_dict['data']['temp']
        w = adlr_dict['data']['w']
        meanr = get_meanr_vs_t_from_cas(adlr_dict, cas_dict, \
                    change_cas_corr, cutoff_bins, incl_rain, \
                    incl_vent)

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        print(np.shape(meanr))
        meanr = meanr[filter_inds]
        print(np.shape(meanr))

        meanr_alldates = add_to_alldates_array(meanr, meanr_alldates)

        make_and_save_meanr_hist(meanr, date)

    make_and_save_meanr_hist(meanr_alldates, 'alldates')

def add_to_alldates_array(meanr, meanr_alldates):

    if meanr_alldates is None:
        return meanr
    else:
        return np.concatenate((meanr_alldates, meanr))

def make_and_save_meanr_hist(meanr, label):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(meanr, bins=30, density=False)
    ax.set_xlabel('meanr (m)')
    ax.set_ylabel('Count')
    ax.set_title(label + ' SS_QSS distb' \
                    + ', change_cas_corr=' + str(change_cas_corr) \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent))
    outfile = FIG_DIR + versionstr + 'meanr_hist_cas_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
