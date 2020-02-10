"""
Create and save figure 0909_meanr_scatter.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.cloudevents_set_figsrc import get_datablock, get_meanr_vs_t
from halo.utils import get_ind_bounds, match_multiple_arrays

matplotlib.rcParams.update({'font.size': 14})

casbinfile = DATA_DIR + 'CAS_bins.npy'
CAS_bins = np.load(casbinfile, allow_pickle=True).item()
cas_centr = (CAS_bins['upper'] + CAS_bins['lower'])/2.
cas_dr = CAS_bins['upper'] - CAS_bins['lower']
cas_nbins = len(cas_centr)

cdpbinfile = DATA_DIR + 'CDP_bins.npy'
CDP_bins = np.load(cdpbinfile, allow_pickle=True).item()
cdp_centr = (CDP_bins['upper'] + CDP_bins['lower'])/2.
cdp_dr = CDP_bins['upper'] - CDP_bins['lower']
cdp_nbins = len(cdp_centr)

def main():
    """
    the main routine.
    """
    dates = ['20140909']
    offsets = [2]

    for m, date in enumerate(dates):
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        casdata['data']['time'] = np.array([t - offsets[m] for t in casdata['data']['time']])
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()
        
        #entire lwc time sequence to plot in every figure for reference
        x3_adlr = adlrdata['data']['time']
        y3_adlr = adlrdata['data']['lwc']
        x3_cas = casdata['data']['time'][casdata['data']['lwc_t_inds']]
        y3_cas = casdata['data']['lwc']['11']
        x3_cdp = cdpdata['data']['time'][cdpdata['data']['lwc_t_inds']]
        y3_cdp = cdpdata['data']['lwc']['11']
        
        #loop through all cloud events and make a figure for each
    [adlrinds, casinds, cdpinds] = match_multiple_arrays(
	[np.around(adlrdata['data']['time']), \
	np.around(casdata['data']['time']), \
	np.around(cdpdata['data']['time'])])
    datablock = get_datablock(adlrinds, casinds, cdpinds, \
	adlrdata, casdata, cdpdata)
    
    #remove rows with error values in any of the three
    goodrows = []
    for i, row in enumerate(datablock):
        if sum(np.isnan(np.concatenate((row[0:2], row[3:])))) == 0:
       	    goodrows.append(i)
    N = len(goodrows)
    Nerr = np.shape(datablock)[0] - N
    datablock = datablock[goodrows, :]
    (cas_meanr, cdp_meanr) = get_meanr_vs_t(datablock)
    fig, ax = plt.subplots()
    ax.scatter(cas_meanr*1.e6, cdp_meanr*1.e6)
   # coef = np.polyfit(cas_meanr, cdp_meanr, 1)
   # print(coef)
   # poly1d_fn = np.poly1d(coef) 
   # ax.plot(cas_meanr, poly1d_fn(cas_meanr), '--k')
    ax.set_xlabel('CAS')
    ax.set_ylabel('CDP')
    ax.set_title('Mean radius (um)')
    ax.set_aspect('equal', 'datalim')
    fig.set_size_inches(21, 12)
    outfile = FIG_DIR + '0909_meanr_scatter_figure.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
