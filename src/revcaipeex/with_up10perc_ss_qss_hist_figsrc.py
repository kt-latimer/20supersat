"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_ss_vs_t, get_lwc

#for plotting
#versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

#cutoff_bins = True 
#incl_rain = False
#incl_vent = False
#full_ss = True

def main():
    
    if len(sys.argv) > 1:
        versionnum = int(sys.argv[1])
        (dummy_bool, cutoff_bins, full_ss, \
            incl_rain, incl_vent) = get_boolean_params(versionnum)
        versionstr = 'v' + str(versionnum) + '_'

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_qss_alldates = None
    up10perc_ss_qss_alldates = None

    for date in good_dates:
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cpd_dict = np.load(cpdfile, allow_pickle=True).item()

        lwc = get_lwc(cpd_dict,cutoff_bins)
        temp = met_dict['data']['temp']
        w = met_dict['data']['w']
        ss_qss = get_ss_vs_t(met_dict, cpd_dict, cutoff_bins, \
                            full_ss, incl_rain, incl_vent)
 
        #there's a weird outlier which the third line removes
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (ss_qss < 100), \
                        (temp > 273)))

        if np.sum(filter_inds) > 0:
            w_filt = w[filter_inds]
            up10perc_cutoff = np.percentile(w_filt, 90)
            up10perc_inds = np.logical_and.reduce((
                            (filter_inds), \
                            (w > up10perc_cutoff)))
            up10perc_ss_qss = ss_qss[up10perc_inds]
        else:
            up10perc_ss_qss = np.array([]) 
            

        print(date)
        ss_qss = ss_qss[filter_inds]

        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)
        up10perc_ss_qss_alldates = \
            add_to_alldates_array(up10perc_ss_qss, up10perc_ss_qss_alldates)

        make_and_save_ss_qss_hist(ss_qss, up10perc_ss_qss, date, versionstr, \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)


    print(ss_qss_alldates)
    print(up10perc_ss_qss_alldates)
    make_and_save_ss_qss_hist(ss_qss_alldates, up10perc_ss_qss_alldates, \
            'alldates', versionstr, cutoff_bins, full_ss, incl_rain, incl_vent)

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def make_and_save_ss_qss_hist(ss_qss, up10perc_ss_qss, label, versionstr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent):
    
    print('# pts total: ' + str(np.sum(ss_qss < 200)))
    if np.sum(ss_qss) != 0:
        print('max: ' + str(np.nanmax(ss_qss)))
        print('# pts ss > 2%: ' + str(np.sum(ss_qss > 2)))
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    #bins = [0+0.7*i for i in range(30)]
    #ax.hist(ss_qss, bins=bins, density=False)
    n, bins, patches = ax.hist(ss_qss, bins=30, density=False, \
            label='All data satisfying filter criteria')
    ax.hist(up10perc_ss_qss, bins=bins, density=False, \
            histtype='stepfilled', linewidth=3, \
            edgecolor='r', facecolor=(0, 0, 0, 0), \
            label='Upper 10th percentile updrafts out of filtered data')
    ax.set_xlabel('SS (%)')
    ax.set_ylabel('Count')
    ax.set_title(label + ' SS_QSS distb' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent) \
                    + ', full_ss=' + str(full_ss))
    plt.legend()
    outfile = FIG_DIR + versionstr + 'with_up10perc_ss_qss_hist_' \
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
