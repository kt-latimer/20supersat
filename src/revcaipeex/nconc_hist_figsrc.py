"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_nconc_vs_t, get_lwc

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

#change_dsd_corr = False
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

    nconc_alldates = None

    for date in good_dates:
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        dsdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        dsd_dict = np.load(dsdfile, allow_pickle=True).item()

        lwc = get_lwc(dsd_dict,cutoff_bins)
        temp = met_dict['data']['temp']
        w = met_dict['data']['w']

        nconc = get_nconc_vs_t(met_dict, dsd_dict, cutoff_bins, \
                                incl_rain, incl_vent)

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        print(date)
        nconc = nconc[filter_inds]

        nconc_alldates = add_to_alldates_array(nconc, nconc_alldates)

        make_and_save_nconc_hist(nconc, date, versionstr, \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)

    print(nconc_alldates)
    make_and_save_nconc_hist(nconc_alldates, 'alldates', versionstr, \
            cutoff_bins, full_ss, incl_rain, incl_vent)

def add_to_alldates_array(nconc, nconc_alldates):

    if nconc_alldates is None:
        return nconc
    else:
        return np.concatenate((nconc_alldates, nconc))

def make_and_save_nconc_hist(nconc, label, versionstr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(nconc, bins=30, density=False)
    ax.set_xlabel('nconc (m^-3)')
    ax.set_ylabel('Count')
    ax.set_title(label + ' cloud drop num conc distb' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent) \
                    + ', full_ss=' + str(full_ss))
    outfile = FIG_DIR + versionstr + 'nconc_hist_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_boolean_params(versionnum):

    versionnum = versionnum - 1 #for modular arithmetic
    
    if versionnum > 23:
        return versionnum

    #here this is a dummy var but keeping for consistency with
    #revhalo for now aka for my sanity
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
