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
                        pad_lwc_arrays, linregress

#for plotting
#colors = {'control': '#777777', 'modified': '#88720A'}
versionstr = 'v4_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5

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
                        (temp > 273)))
        
        #apply lwc filters
        w = w[filter_inds]

        n_tot = w.shape[0]
        n_hi_w = np.sum(w > 7)

        if i == 0:
            w_alldates = w
        else:
            w_alldates = np.concatenate((w_alldates, w))

        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(w, bins=30, log=True)#, color=colors['tot_derv'])
        ax.text(0.9, 0.9, '$N_{total}$: ' + str(n_tot), transform=ax.transAxes) 
        ax.text(0.9, 0.8, '$N_{> 7 m/s}$: ' + str(n_hi_w), transform=ax.transAxes) 
        #ax.set_title('w distribution, no filter')
        #ax.set_title('w distribution, LWC > 1e-5 g/g')
        ax.set_title('w distribution, LWC > 1e-5 g/g, T > 273K')
        #ax.set_title('SS distribution, w > 2 m/s')
        #ax.set_title('SS distribution, LWC > 1e-5 g/g, w > 2 m/s')
        ax.set_xlabel('w (m/s)')
        ax.set_ylabel('Count')
        outfile = FIG_DIR + versionstr + 'w_from_adlr_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

    #make histogram
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(w_alldates, bins=30, log=True)#, color=colors['tot_derv'])
    n_tot = w_alldates.shape[0]
    n_hi_w = np.sum(w_alldates > 7)
    ax.text(0.9, 0.9, '$N_{total}$: ' + str(n_tot), transform=ax.transAxes) 
    ax.text(0.9, 0.8, '$N_{> 7 m/s}$: ' + str(n_hi_w), transform=ax.transAxes) 
    #ax.set_title('w distribution, no filter')
    #ax.set_title('w distribution, LWC > 1e-5 g/g')
    ax.set_title('w distribution, LWC > 1e-5 g/g, T > 273K')
    #ax.set_title('SS distribution, LWC > 1e-5 g/g')
    #ax.set_title('SS distribution, w > 2 m/s')
    #ax.set_title('SS distribution, LWC > 1e-5 g/g, w > 2 m/s')
    ax.set_xlabel('w (m/s)')
    ax.set_ylabel('Count')
    outfile = FIG_DIR + versionstr + 'w_from_adlr_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
