
"""
some module to do things.
"""

import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_nconc_vs_t, \
                        get_meanr_vs_t, linregress

nconc_filter_val = 10.e6
meanr_filter_val = 1.e-6

def main():
    """
    the main routine.
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

        r_squared = []            
        
        orig_time = casdata['data']['time']
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

            (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, True, True)
            (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, True, True)
            filter_inds = np.logical_and.reduce((
                            (nconc_cas > nconc_filter_val), \
                            (nconc_cdp > nconc_filter_val), \
                            (meanr_cas > meanr_filter_val), \
                            (meanr_cdp > meanr_filter_val)))

            #apply num conc and mean radius filters and change to units of ccm
            meanr_cas = meanr_cas[filter_inds]*1.e6
            meanr_cdp = meanr_cdp[filter_inds]*1.e6

            #get linear regression params
            m_nconc, b_nconc, R_nconc, sig_nconc = linregress(nconc_cas, nconc_cdp)
            m_meanr, b_meanr, R_meanr, sig_meanr = linregress(meanr_cas,
            meanr_cdp)
            print(offset, R_nconc**2, R_meanr**2, m_nconc, m_meanr)

if __name__ == "__main__":
    main()
