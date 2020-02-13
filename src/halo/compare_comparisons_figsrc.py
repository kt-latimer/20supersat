""" 
Scrap code for trying to figure out discrepancies between current code and
previous Matlab code.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import match_multiple_arrays, get_datablock, \
                        get_nconc_vs_t, get_meanr_vs_t

matplotlib.rcParams.update({'font.size': 22})

nconc_filter_val = 10.e6
meanr_filter_val = 1.e-6

def main():
    """
    the main routine.
    """
    date = '20140909'
    offset = 2

    #from matlab code
    qz_time = np.loadtxt(DATA_DIR + 'time.txt')
    qz_nconc_cas = np.loadtxt(DATA_DIR + 'nconc_cas.txt')
    qz_nconc_cas_corr = np.loadtxt(DATA_DIR + 'nconc_cas_corr.txt')
    qz_meanr_cas = np.loadtxt(DATA_DIR + 'meanr_cas.txt')
    qz_nconc_cdp = np.loadtxt(DATA_DIR + 'nconc_cdp.txt')
    qz_meanr_cdp = np.loadtxt(DATA_DIR + 'meanr_cdp.txt')
    
    #load data
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlrdata = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    casdata = np.load(casfile, allow_pickle=True).item()
    casdata['data']['time'] = \
        np.array([t - offset for t in casdata['data']['time']])
    cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdpdata = np.load(cdpfile, allow_pickle=True).item()
    
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


    [tinds_qz, tinds_me] = match_multiple_arrays([np.around(qz_time), \
        np.around(datablock[:, 0])])
    print(qz_time[tinds_qz][0:100])
    print()
    #get time-aligned nconc and meanr data for comparison of correction
    #schemes
    (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, False, False)
    (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, False, False)
    print(datablock[tinds_me[0:100], 0])
    filter_inds = np.logical_and.reduce((
                    (nconc_cas > nconc_filter_val), \
                    (nconc_cdp > nconc_filter_val), \
                    (meanr_cas > meanr_filter_val), \
                    (meanr_cdp > meanr_filter_val)))

    fig, ax = plt.subplots()
    #ax.scatter(qz_nconc_cas[tinds_qz], nconc_cas[tinds_me])
    ax.scatter(qz_nconc_cas[[tind+1 for tind in tinds_qz]], 1.e6*nconc_cas[tinds_me])

    outfile = FIG_DIR + 'compare_comparisons_figure2.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
