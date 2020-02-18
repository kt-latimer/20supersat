"""
Calculate linear regression parameters for CAS vs CDP in number concentration
and mean radius measured values. (Just prints everything out)
"""

import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_meanr_vs_t, \
                        get_nconc_vs_t, linregress, high_bin_cas, \
                        high_bin_cdp, pad_lwc_arrays

lwc_filter_val = 1.e-5
meanr_filter_val = 1.e-6
nconc_filter_val = 10.e6

change_cas_corr = True
cutoff_bins = True

def main():
    """
    For each date with data from all three instruments, fit CAS vs CDP in
    number concentration and mean radius to a line and for each time offset,
    print out R_nconc^2, R_meanr^2, m_nconc, m_meanr
    """
    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
         '20140919', '20140918', '20140921', '20140927', '20140928', \
         '20140930', '20141001']

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

        orig_time = casdata['data']['time']
        #loop through reasonable time offset range ($\pm$ 9 sec)
        for offset in range(-9, 9):
            casdata['data']['time'] = \
                np.array([t - offset for t in orig_time])
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

            #get time-aligned nconc and meanr data 
            (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, change_cas_corr,
                                                    cutoff_bins)
            (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, change_cas_corr,
                                                    cutoff_bins)

            ##filter out low values (have not done any sensitivity analysis for
            ##these parameters)
            #filter_inds = np.logical_and.reduce((
            #                (nconc_cas > nconc_filter_val), \
            #                (nconc_cdp > nconc_filter_val), \
            #                (meanr_cas > meanr_filter_val), \
            #                (meanr_cdp > meanr_filter_val)))
            
            ##filter on LWC vals
            booleanind = int(change_cas_corr) + int(cutoff_bins)*2
            lwc_cas = datablock[:, high_bin_cas+booleanind]
            lwc_cdp = datablock[:, high_bin_cdp+booleanind]
            filter_inds = np.logical_and.reduce((
                            (lwc_cas > lwc_filter_val), \
                            (lwc_cdp > lwc_filter_val), \
                            np.logical_not(np.isnan(nconc_cas)), \
                            np.logical_not(np.isnan(nconc_cdp))))

            #apply mean rad / num conc filters and change to units of um / ccm
            meanr_cas = meanr_cas[filter_inds]*1.e6
            meanr_cdp = meanr_cdp[filter_inds]*1.e6
            nconc_cas = nconc_cas[filter_inds]*1.e-6
            nconc_cdp = nconc_cdp[filter_inds]*1.e-6

            #get linear regression params
            m_nconc, b_nconc, R_nconc, sig_nconc = linregress(nconc_cas, nconc_cdp)
            m_meanr, b_meanr, R_meanr, sig_meanr = linregress(meanr_cas, meanr_cdp)
            print(offset, R_nconc**2, R_meanr**2, m_nconc, m_meanr)

if __name__ == "__main__":
    main()
