"""
Calculate linear regression parameters for TAS reported values from CAS vs
ADLR. (Just prints everything out)
"""

import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        pad_lwc_arrays, linregress

lwc_filter_val = 1.e-5

change_cas_corr = True
cutoff_bins = True

def main():
    """
    For each date with data from all three instruments, fit ADLR vs CAS TAS
    measured values, and for each value of ADLR/CAS time offset, print out
    R_tas^2, m_tas, b_tas.
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

        orig_time_adlr = adlrdata['data']['time']
        orig_time_cas = casdata['data']['time']
        #loop through reasonable time offset range ($\pm$ 9 sec)
        for offset in range(-9, 9):
            adlrdata['data']['time'] = \
                np.array([t + offset for t in orig_time_adlr])
            #casdata['data']['time'] = \
            #    np.array([t - offset for t in orig_time_cas])
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

            #get time-aligned TAS data 
            abs_w_adlr = np.abs(datablock[:, 2])

            #filter on LWC vals
            booleanind = int(change_cas_corr) + int(cutoff_bins)*2
            lwc_cas = datablock[:, high_bin_cas+booleanind]
            filter_inds = np.logical_and.reduce((
                            (lwc_cas > lwc_filter_val), \
                            np.logical_not(np.isnan(abs_w_adlr))))

            #apply lwc filters
            abs_w_adlr = abs_w_adlr[filter_inds]
            lwc_cas = lwc_cas[filter_inds]

            #get linear regression params
            m, b, R, sig = linregress(abs_w_adlr, lwc_cas)
            print(offset, R**2, m, b)

if __name__ == "__main__":
    main()
