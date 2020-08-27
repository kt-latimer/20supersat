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
                        match_multiple_arrays, low_bin_cas, high_bin_cas, \
                        pad_lwc_arrays, linregress, get_full_ss_vs_t, \
                        get_meanr_vs_t, get_nconc_vs_t, centr_cas, dr_cas, \
                        low_bin_cdp, high_bin_cdp, centr_cdp, dr_cdp

#for plotting
colors = {'ss': '#88720A'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
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

        #get time-aligned theta data 
        temp = datablock[:, 1]
        time = datablock[:, 0]
        w = datablock[:, 2]

        #filter on LWC vals
        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock[:, high_bin_cas+booleanind]
        #lwc_cdp = datablock[:, high_bin_cdp+booleanind]
        #filter_inds = w > 2
        #filter_inds = lwc_cas > lwc_filter_val
        #filter_inds = np.logical_and.reduce((
        #                (lwc_cas > lwc_filter_val), \
        #                (w > 2)))
        filter_inds = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        #(lwc_cdp > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))
        
        N = np.sum(filter_inds)
        delta_rgb = 200./N
        greyscale_colors = [((20 + i*delta_rgb)/255., (20 + i*delta_rgb)/255., \
                            (20 + i*delta_rgb)/255.) for i in range(N)] 
        print(greyscale_colors)
        print(len(greyscale_colors))

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(24, 12)
        ngraphed = 0
        for j, val in enumerate(filter_inds):
            if val:
                #units cm^-3 um^-1
                dNdr = 1.e-12*np.array([datablock[j, k] for k in \
                       range(low_bin_cas, high_bin_cas)])/dr_cas 
                dNdlnr = dNdr*centr_cas
                ax[0].plot(centr_cas*1.e6, dNdr, marker=None, \
                            linewidth=1, color=greyscale_colors[ngraphed])
                ax[1].plot(centr_cas*1.e6, dNdlnr, marker=None, \
                            linewidth=1, color=greyscale_colors[ngraphed])
                #dNdr = 1.e-12*np.array([datablock[j, k] for k in \
                #       range(low_bin_cdp, high_bin_cdp)])/dr_cdp
                #dNdlnr = dNdr*centr_cdp
                #ax[0].plot(centr_cdp*1.e6, dNdr, \
                #            marker=None, linewidth=3, label=time[j])
                #ax[1].plot(centr_cdp*1.e6, dNdlnr, \
                #            marker=None, linewidth=3, label=time[j])
                ngraphed += 1

        ax[0].set_xlabel(r'r ($\mu$m)')
        ax[0].set_ylabel(r'dN/dr (cm$^{-3}$ $\mu$m$^{-1}$)')
        ax[1].set_xlabel(r'r ($\mu$m)')
        ax[1].set_ylabel(r'dN/d(ln(r)) (cm$^{-3}$)')
        fig.suptitle(date + ' LWC > 1.e-4 g/g, T > 273K, w > 2 m/s')

        outfile = FIG_DIR + versionstr + 'greyscale_indiv_dsds_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

if __name__ == "__main__":
    main()
