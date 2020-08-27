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
                        pad_lwc_arrays, linregress, get_full_ss_vs_t

#for plotting
colors = {'ss': '#88720A'}
versionstr = 'v3_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 2.e-4
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
        w = datablock[:, 2]
        temp = datablock[:, 1]
        (ss_cas, ss_cdp) = \
            get_full_ss_vs_t(datablock, change_cas_corr, cutoff_bins)
        print(np.shape(ss_cas))
        print(ss_cas)

        #filter on LWC vals
        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock[:, high_bin_cas+booleanind]
        #filter_inds = w > 2
        #filter_inds = lwc_cas > lwc_filter_val
        #filter_inds = np.logical_and.reduce((
        #                (lwc_cas > lwc_filter_val), \
        #                (w > 2)))
        filter_inds = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))
        
        #apply lwc filters
        ss_cas = ss_cas[filter_inds]
        print(np.shape(ss_cas))
        print(ss_cas)

        if i == 0:
            ss_alldates = ss_cas
        else:
            ss_alldates = np.concatenate((ss_alldates, ss_cas))
        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(ss_cas, bins=30)#, color=colors['tot_derv'])
        #ax.set_title('w distribution, no filter')
        #ax.set_title('w distribution, LWC > 1e-5 g/g')
        ax.set_title('SS distribution, LWC > 2.e-4 g/g, T > 273K, w > 2 m/s')
        #ax.set_title('SS distribution, w > 2 m/s')
        #ax.set_title('SS distribution, LWC > 1e-5 g/g, w > 2 m/s')
        ax.set_xlabel('SS (%)')
        ax.set_ylabel('Count')
        outfile = FIG_DIR + versionstr + 'ss_from_cas_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

    #make histogram
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    for thing in ss_alldates:
        print(thing)
    ax.hist(ss_alldates, bins=30)#, color=colors['tot_derv'])
    #ax.set_title('w distribution, no filter')
    #ax.set_title('w distribution, LWC > 1e-5 g/g')
    ax.set_title('SS distribution, LWC > 2.e-4 g/g, T > 273K, w > 2 m/s')
    #ax.set_title('SS distribution, LWC > 1e-5 g/g')
    #ax.set_title('SS distribution, w > 2 m/s')
    #ax.set_title('SS distribution, LWC > 1e-5 g/g, w > 2 m/s')
    ax.set_xlabel('SS (%)')
    ax.set_ylabel('Count')
    outfile = FIG_DIR + versionstr + 'ss_from_cas_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
