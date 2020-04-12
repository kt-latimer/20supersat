"""
Calculate linear regression parameters for derived QSS SS values from CAS vs
CDP. (Just prints everything out)
"""

import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        high_bin_cdp, pad_lwc_arrays, linregress, \
                        get_ss_vs_t

lwc_filter_val = 1.e-5

change_cas_corr = True
cutoff_bins = True

def main():
    """
    For each date with data from all three instruments, fit CAS vs CDP
    (adjusted to optimal offset as determined by optimal_offsets.py) in
    qss ss to a line and for each adlr/cas time offset, print out R_ss^2,
    m_ss, b_ss
    """
    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
            '20140918', '20140921', '20140927', '20140928', \
            '20140930', '20141001']
    optimal_cascdp_offsets = [3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 2]

    for i, date in enumerate(dates):
        print(date)
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata, change_cas_corr, cutoff_bins)
        cdpdata = pad_lwc_arrays(cdpdata, change_cas_corr, cutoff_bins)

        #original times
        orig_time_adlr = adlrdata['data']['time']
        orig_time_cas = casdata['data']['time']

        #implement offset that optimizes lwc agreement between cas and cdp
        casdata['data']['time'] = \
            np.array([t - optimal_cascdp_offsets[i] for t in orig_time_cas])

        #loop through reasonable time offset range ($\pm$ 9 sec)
        for offset in range(-9, 9):
            #offset adlr from cas, which has also been offset
            adlrdata['data']['time'] = \
                np.array([t - optimal_cascdp_offsets[i] + offset for t in orig_time_adlr])
            #align all datasets along time.set_aspect
            [adlrinds, casinds, cdpinds] = match_multiple_arrays(
                [np.around(adlrdata['data']['time']), \
                np.around(casdata['data']['time']), \
                np.around(cdpdata['data']['time'])])
            datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                        adlrdata, casdata, cdpdata)

            #remove rows with error values (except vert wind vel cause it's shit)
            goodrows = []
            for j, row in enumerate(datablock):
                if sum(np.isnan(np.concatenate(((row[0:2], row[3:]))))) == 0:
                    goodrows.append(j)
            datablock = datablock[goodrows, :]

            #get time-aligned QSS SS data 
            (ss_cas, ss_cdp) = get_ss_vs_t(datablock, \
                                            change_cas_corr, cutoff_bins)
            #print(ss_cas)
            #print(ss_cdp)
            #print(np.nanmean(ss_cas))
            #print(np.nanstd(ss_cas))
            #print(np.nanmean(ss_cdp))
            #print(np.nanstd(ss_cdp))

            #filter on LWC vals
            booleanind = int(change_cas_corr) + int(cutoff_bins)*2
            lwc_cas = datablock[:, high_bin_cas+booleanind]
            lwc_cdp = datablock[:, high_bin_cdp+booleanind]
            filter_inds = np.logical_and.reduce((
                            (lwc_cas > lwc_filter_val), \
                            (lwc_cdp > lwc_filter_val), \
                            np.logical_not(np.isnan(ss_cas)), \
                            np.logical_not(np.isnan(ss_cdp))))

            #apply lwc filters
            ss_cas = ss_cas[filter_inds]
            ss_cdp = ss_cdp[filter_inds]

            #get ss stdev and 99th percentile
            print(offset, np.std(ss_cas), np.percentile(ss_cas, 99))

            #get linear regression params
            #m_ss, b_ss, R_ss, sig_ss = linregress(ss_cas, ss_cdp)
            #print(offset, R_ss**2, m_ss, b_ss)

if __name__ == "__main__":
    main()
