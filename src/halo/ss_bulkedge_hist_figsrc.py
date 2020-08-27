"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        pad_lwc_arrays, linregress, \
                        get_full_ss_vs_t

#for plotting
colors = {'bulk': '#095793', 'edge': '#88720A'}
versionstr = 'v3_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5
w_cutoff = 2

change_cas_corr = True
cutoff_bins = True

def main():
    """
    for each date and for all dates combined, create and save w histogram.
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

        #loop through reasonable time offset range ($\pm$ 9 sec)
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(casdata['data']['time']), \
            np.around(cdpdata['data']['time'])])
        datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]

        #get env and supersat data 
        alt = datablock[:, -1]
        temp = datablock[:, 1]
        w = datablock[:, 2]
        (ss_cas, ss_cdp) = \
            get_full_ss_vs_t(datablock, change_cas_corr, cutoff_bins)

        #get lwc data
        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock[:, high_bin_cas+booleanind]

        #set up arrays
        nperlayer = []
        nedgeperlayer = []
        lwcfifthperc = []
        ssbulk = []
        ssedge = []
        layermax = 500 #m

        #for taking 5th perc of entire set
        total_filter = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        (w > w_cutoff), \
                        (w < 100), \
                        (temp > 273)))
        
        if np.sum(total_filter) != 0:
            total_cutoff = np.percentile(lwc_cas[total_filter], 5)
            bulk_filter = lwc_cas[total_filter] >= total_cutoff
            edge_filter = np.logical_not(bulk_filter)
            ssbulk = ss_cas[total_filter][bulk_filter]
            ssedge = ss_cas[total_filter][edge_filter]
        else:
            ssbulk = []
            ssedge = []
        ##group data in 500m layers by altitude
        #while layermax < np.max(alt):
        #    layer_filter = np.logical_and.reduce((
        #                    (lwc_cas > lwc_filter_val), \
        #                    (layermax-500 <= alt), \
        #                    (alt < layermax), \
        #                    (w > w_cutoff), \
        #                    (temp > 273)))
        #    nedge = 0
        #    if np.sum(layer_filter) != 0:
        #        nperlayer.append(np.sum(layer_filter))
        #        perc_cutoff = np.percentile(lwc_cas[layer_filter], 5)
        #        lwcfifthperc.append(perc_cutoff)
        #        for j, val in enumerate(ss_cas[layer_filter]):
        #            if lwc_cas[layer_filter][j] < perc_cutoff:
        #                nedge += 1
        #                ssedge.append(val)
        #            else:
        #                ssbulk.append(val)
        #    else:
        #        nperlayer.append(0)
        #        lwcfifthperc.append(np.nan)
        #    nedgeperlayer.append(nedge)
        #    layermax += 500
        
        print(nperlayer)
        print(nedgeperlayer)
        print(lwcfifthperc)
        print(ssedge)
        
        if i == 0:
            ssbulk_alldates = ssbulk
            ssedge_alldates = ssedge
        else:
            ssbulk_alldates = np.concatenate((ssbulk_alldates, ssbulk))
            ssedge_alldates = np.concatenate((ssedge_alldates, ssedge)) 

        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(ssbulk, bins=30, color=colors['bulk'], alpha=0.7, label='bulk')
        ax.hist(ssedge, bins=30, color=colors['edge'], alpha=0.7, label='edge')
        ax.set_title('SS distribution, LWC > 1.e-5 g/g, T > 273K, w > 2 m/s')
        ax.set_xlabel('SS (%)')
        ax.set_ylabel('Count')
        ax.legend(loc=1)
        outfile = FIG_DIR + versionstr + 'ss_bulkedge_hist_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

    #make histogram
    print(ssbulk_alldates.shape)
    print(ssedge_alldates.shape)
    nbulk = np.sum(ssbulk_alldates >= 2)
    nedge = np.sum(ssedge_alldates >= 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(ssbulk_alldates, bins=30, color=colors['bulk'], alpha=0.7, label='bulk')
    ax.hist(ssedge_alldates, bins=30, color=colors['edge'], alpha=0.7, label='edge')
    ax.text(0.8, 0.8, '$N_{SS>2\%, bulk}$: ' + str(nbulk), transform=ax.transAxes) 
    ax.text(0.8, 0.7, '$N_{SS>2\%, edge}$: ' + str(nedge), transform=ax.transAxes) 
    ax.set_title('SS distribution, LWC > 1.e-5 g/g, T > 273K, w > 2 m/s')
    ax.set_xlabel('SS (%)')
    ax.set_ylabel('Count')
    ax.legend(loc=1)
    outfile = FIG_DIR + versionstr + 'ss_bulkedge_hist_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
