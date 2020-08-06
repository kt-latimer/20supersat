"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock_with_cip, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas_with_cip, \
                        pad_lwc_arrays_with_cip, linregress, \
                        get_full_ss_vs_t_with_cip, get_meanfr_vs_t_with_cip, \
                        get_nconc_vs_t_with_cip, get_full_ss_vs_t_with_cip_and_vent

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
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cipdata = np.load(cipfile, allow_pickle=True).item()

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays_with_cip(casdata, change_cas_corr, cutoff_bins)
        cipdata = pad_lwc_arrays_with_cip(cipdata, change_cas_corr, cutoff_bins)

        #loop through reasonable time offset range ($\pm$ 9 sec)
        [adlrinds, casinds, cipinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(casdata['data']['time']), \
            np.around(cipdata['data']['time'])])
        datablock = get_datablock_with_cip(adlrinds, casinds, cipinds, \
                                    adlrdata, casdata, cipdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]
        meanfr = get_meanfr_vs_t_with_cip(datablock, True, True)
        nconc = get_nconc_vs_t_with_cip(datablock, True, True)

        #get time-aligned theta data 
        w = datablock[:, 2]
        temp = datablock[:, 1]
        ss = get_full_ss_vs_t_with_cip_and_vent(datablock, change_cas_corr, cutoff_bins)
        ss2 = get_full_ss_vs_t_with_cip(datablock, change_cas_corr, cutoff_bins)

        #filter on LWC vals
        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock[:, high_bin_cas_with_cip+booleanind]
        #filter_inds = w > 2
        #filter_inds = lwc_cas > lwc_filter_val
        #filter_inds = np.logical_and.reduce((
        #                (lwc_cas > lwc_filter_val), \
        #                (w > 2)))
        filter_inds = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))
        #print(w[filter_inds])
        #print(meanr[filter_inds])
        #print(nconc[filter_inds])
        ##apply lwc filters
        ss = ss[filter_inds]
        ss2 = ss2[filter_inds]
        #print(ss)
        #print(ss2)
        
        tau = 1./(meanfr*nconc)
        meanfr = meanfr[filter_inds]
        #print(np.mean(meanfr[filter_inds]), np.std(meanfr[filter_inds]), \
        #        np.median(meanfr[filter_inds]), np.min(meanfr[filter_inds]), \
        #        np.max(meanfr[filter_inds]))
        #print(np.mean(meanfr[filter_inds]), np.std(meanfr[filter_inds]), \
        #        np.min(meanfr[filter_inds]), np.max(meanfr[filter_inds]))
        #print(np.mean(nconc[filter_inds]), np.std(nconc[filter_inds]), \
        #        np.min(nconc[filter_inds]), np.max(nconc[filter_inds]))

        if i == 0:
            meanfr_alldates = meanfr
        else:
            meanfr_alldates = np.concatenate((meanfr_alldates, meanfr))
        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(meanfr, bins=30)#, color=colors['tot_derv'])
        #ax.set_title('w distribution, no filter')
        #ax.set_title('w distribution, LWC > 1e-5 g/g')
        ax.set_title('meanfr distribution, LWC > 1.e-4 g/g, T > 273K, w > 2 m/s, \
        with raindrops and ventilation corrections')
        #ax.set_title('meanfr distribution, w > 2 m/s')
        #ax.set_title('meanfr distribution, LWC > 1e-5 g/g, w > 2 m/s')
        ax.set_xlabel('meanfr (m)')
        ax.set_ylabel('Count')
        outfile = FIG_DIR + versionstr + 'meanfr_with_cip_from_cas_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

    #make histogram
    print(meanfr_alldates.shape)
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(meanfr_alldates, bins=30)#, color=colors['tot_derv'])
    #ax.set_title('w distribution, no filter')
    #ax.set_title('w distribution, LWC > 1e-5 g/g')
    ax.set_title('meanfr distribution, LWC > 1.e-4 g/g, T > 273K, w > 2 m/s, \
        with raindrops and ventilation corrections')
    #ax.set_title('meanfr distribution, LWC > 1e-5 g/g')
    #ax.set_title('meanfr distribution, w > 2 m/s')
    #ax.set_title('meanfr distribution, LWC > 1e-5 g/g, w > 2 m/s')
    ax.set_xlabel('meanfr (m)')
    ax.set_ylabel('Count')
    outfile = FIG_DIR + versionstr + 'meanfr_with_cip_from_cas_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
