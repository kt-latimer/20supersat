"""
Create and save nconc_vs_lwc figure set. 
"""
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_nconc_vs_t, \
                        get_nconc_vs_t, linregress, high_bin_cas, \
                        high_bin_cdp

#for plotting
colors = {'ADLR': '#777777', 'CAS': '#95B9E9', 'CDP': '#FC6A0C', 'w': '#BA3F00'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

change_cas_corr = True
cutoff_bins = True

def main():
    """
    For each date, and also for aggregated data from all dates, make a 
    scatter plot of LWC vs particle num concentration - one for CAS, one for CDP.
    Use all three correction schemes (i.e. 3um bin cutoff, change CAS vol corr
    factor, and use optimal time offsets)
    """

    dates = ['20140906', '20140909', '20140911', '20140912', \
            '20140916', '20140918', '20140921', '20140927', \
            '20140928', '20140930', '20141001']
    offsets = [3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 2]
    
    for i, date in enumerate(dates):
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        castime = casdata['data']['time']
        castime_offset = [t - offsets[i] for t in castime]
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()
        
        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata)
        cdpdata = pad_lwc_arrays(cdpdata)

        #datablock with time offsets 
        #align all datasets along time.set_aspect
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(castime_offset), \
            np.around(cdpdata['data']['time'])])
        datablock_offset = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock_offset):
            if sum(np.isnan(np.concatenate(((row[0:2], row[3:]))))) == 0:
                goodrows.append(j)
        datablock_offset = datablock_offset[goodrows, :]

        (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock_offset, change_cas_corr,
                                                cutoff_bins)
        nconc_cas = nconc_cas*1.e-6
        nconc_cdp = nconc_cdp*1.e-6

        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock_offset[:, high_bin_cas+booleanind]
        lwc_cdp = datablock_offset[:, high_bin_cdp+booleanind]

        make_figure(nconc_cas, nconc_cdp, lwc_cas, lwc_cdp, date)

        #aggregate data set
        if i == 0:
            all_dates_nconc_cas = nconc_cas
            all_dates_nconc_cdp = nconc_cdp
            all_dates_lwc_cas = lwc_cas
            all_dates_lwc_cdp = lwc_cdp
        else:
            all_dates_nconc_cas = np.concatenate((
                                        all_dates_nconc_cas, nconc_cas))
            all_dates_nconc_cdp = np.concatenate((
                                        all_dates_nconc_cdp, nconc_cdp))
            all_dates_lwc_cas = np.concatenate((
                                        all_dates_lwc_cas, lwc_cas))
            all_dates_lwc_cdp = np.concatenate((
                                        all_dates_lwc_cdp, lwc_cdp))

    #make figures for all dates
    make_figure(all_dates_nconc_cas, all_dates_nconc_cdp, all_dates_lwc_cas,
                all_dates_lwc_cdp, 'alldates')

    #blank figure for make compatibility
    fig, ax = plt.subplots()
    outfile = FIG_DIR + 'nconc_vs_lwc_set_figure.png'
    fig.savefig(outfile)

def make_figure(nconc_cas, nconc_cdp, lwc_cas, lwc_cdp, date):
    """
    Make scatter plot of nconc vs lwc for both instruments (same axis)
    """
    #get limits of the data for plotting purposes
    xlim = np.nanmax(np.array( \
                    [np.nanmax(nconc_cas), \
                    np.nanmax(nconc_cdp)]))

    ylim = np.nanmax(np.array( \
                    [np.nanmax(lwc_cas), \
                    np.nanmax(lwc_cdp)]))

    x_lims = np.array([0, xlim])
    y_lims = np.array([0, ylim])

    #both correction schemes applied
    fig, ax = plt.subplots()
    ax.scatter(nconc_cas, lwc_cas, \
                    c=colors['CAS'], \
                    label='CAS')
    ax.scatter(nconc_cdp, lwc_cdp, \
                    c=colors['CDP'], \
                    label='CDP')
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_xlabel('Number concentration (cm^-3)')
    ax.set_ylabel('LWC (g/g)')
    handles, labels = ax.get_legend_handles_labels()

    #order set manually based on what matplotlib outputs by default
    fig.legend()
    fig.set_size_inches(21, 12)
    figtitle = 'Number concentration cm^-3) vs LWC, Date: ' + date
    fig.suptitle(figtitle)
    outfile = FIG_DIR + versionstr + 'nconc_vs_lwc_' + date + '.png'
    fig.savefig(outfile)
    plt.close()

def pad_lwc_arrays(dataset):
    lwc_t_inds = dataset['data']['lwc_t_inds']
    dataset_shape = np.shape(dataset['data']['time'])

    for cutoff_bins, change_cas_corr in product([True, False], repeat=2):
        booleankey = str(int(cutoff_bins)) \
            + str(int(change_cas_corr)) 
        padded_arr = np.empty(dataset_shape)
        padded_arr[:] = np.nan
        lwc_vals = dataset['data']['lwc'][booleankey]
        padded_arr[lwc_t_inds] = lwc_vals
        dataset['data']['lwc'][booleankey] = padded_arr

    return dataset

if __name__ == "__main__":
    main()
