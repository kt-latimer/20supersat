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
                        pad_lwc_arrays, linregress, get_full_ss_vs_t, \
                        get_nconc_vs_t

#for plotting
colors = {'ss': '#88720A'}
versionstr = 'v2_'

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
        lwc_cas = casdata['data']['lwc_calc'][casinds]/1.e6
        nconc_cas = casdata['data']['nconc_tot_TAS_corr'][casinds]
        temp = adlrdata['data']['stat_temp'][adlrinds]
        w = adlrdata['data']['vert_wind_vel'][adlrinds]
        datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)
        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]
        lwc_cas = lwc_cas[goodrows]
        nconc_cas = nconc_cas[goodrows]
        temp = temp[goodrows]
        w = w[goodrows]

        filter_inds = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))
        
        nconc_cas = nconc_cas[filter_inds]

        if i == 0:
            nconc_alldates = nconc_cas
        else:
            nconc_alldates = np.concatenate((nconc_alldates, nconc_cas))
        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(nconc_cas, bins=30)#, color=colors['tot_derv'])
        #ax.set_title('w distribution, no filter')
        #ax.set_title('w distribution, LWC > 1e-5 g/g')
        ax.set_title('nconc distribution, LWC > 2.e-4 g/g, T > 273K, w > 2 m/s')
        #ax.set_title('nconc distribution, w > 2 m/s')
        #ax.set_title('nconc distribution, LWC > 1e-5 g/g, w > 2 m/s')
        ax.set_xlabel('nconc (m^-3)')
        ax.set_ylabel('Count')
        outfile = FIG_DIR + versionstr + 'nconc_from_ames_cas_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

    #make histogram
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    for thing in nconc_alldates:
        print(thing)
    ax.hist(nconc_alldates, bins=30)#, color=colors['tot_derv'])
    #ax.set_title('w distribution, no filter')
    #ax.set_title('w distribution, LWC > 1e-5 g/g')
    ax.set_title('nconc distribution, LWC > 2.e-4 g/g, T > 273K, w > 2 m/s')
    #ax.set_title('nconc distribution, LWC > 1e-5 g/g')
    #ax.set_title('nconc distribution, w > 2 m/s')
    #ax.set_title('nconc distribution, LWC > 1e-5 g/g, w > 2 m/s')
    ax.set_xlabel('nconc (m^-3)')
    ax.set_ylabel('Count')
    outfile = FIG_DIR + versionstr + 'nconc_from_ames_cas_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
