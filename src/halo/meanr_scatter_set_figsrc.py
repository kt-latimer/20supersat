"""
Create and save figure meanr_scatter_set. This is a significant refactoring
relative to the previous version of the code, now that I have a better idea
what is going on with the data. In particular, condense plotting into a single
generic routine to elminate redundancy.

warning: notation with line breaks is inconsistent.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_meanr_vs_t, \
                        get_nconc_vs_t, linregress, high_bin_cas, \
                        high_bin_cdp, pad_lwc_arrays

#for plotting
colors = {'control': '#777777', 'modified': '#4A8CCA'}
versionstr = 'v11_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 3.e-5
meanr_filter_val = 0#1.e-6
nconc_filter_val = 1.5e6

change_cas_corr = True
cutoff_bins = True

def main():
    """
    For each date, and also for aggregated data from all dates, make four
    scatter plots of cloud droplet mean readius values measured by CAS vs CDP:
    control vs with time offset; control vs correct CAS with xi factor; control
    vs cutoff bins with particle diameters below 3um; control vs all three of
    those correction schemes.
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
        cdptime = cdpdata['data']['time']
        cdptime_offset = [t + offsets[i] for t in cdptime]

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata, change_cas_corr, cutoff_bins)
        cdpdata = pad_lwc_arrays(cdpdata, change_cas_corr, cutoff_bins)

        #datablock without time offsets (ideally the function would have some
        #boolean parameter for this but not a priority for now)
        #align all datasets along time
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(castime), \
            np.around(cdptime)])
        datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(np.concatenate(((row[0:2], row[3:]))))) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]

        #datablock with time offsets 
        #align all datasets along time.set_aspect
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(castime), \
            np.around(cdptime_offset)])
        datablock_offset = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock_offset):
            if sum(np.isnan(np.concatenate(((row[0:2], row[3:]))))) == 0:
                goodrows.append(j)
        datablock_offset = datablock_offset[goodrows, :]

        #make figures for this date
        make_comparison_scatter(datablock, datablock_offset, False, False,
            'Optimal time offset', '_toffset', date)
        make_comparison_scatter(datablock, datablock, True, False, 
            'Same volumetric rescaling factors', '_cascorr', date)
        make_comparison_scatter(datablock, datablock, False, True, 
            '3um lower bin cutoff', '_cutoff', date)
        make_comparison_scatter(datablock, datablock_offset, True, True, 
            'All corrections applied', '_allmod', date)
        
        #aggregate data set
        if i == 0:
            all_dates_datablock = datablock
            all_dates_datablock_offset = datablock_offset
        else:
            all_dates_datablock = \
                np.concatenate((all_dates_datablock, datablock))
            all_dates_datablock_offset = \
                np.concatenate((all_dates_datablock_offset, datablock_offset))

    #make figures for all dates
    make_comparison_scatter(all_dates_datablock, all_dates_datablock_offset, False, False,
        'Optimal time offset', '_toffset', 'all')
    make_comparison_scatter(all_dates_datablock, all_dates_datablock, True, False, 
        'Same volumetric rescaling factors', '_cascorr', 'all')
    make_comparison_scatter(all_dates_datablock, all_dates_datablock, False, True, 
        '3um lower bin cutoff', '_cutoff', 'all')
    make_comparison_scatter(all_dates_datablock, all_dates_datablock_offset, True, True, 
        'All corrections applied', '_allmods', 'all')

    #blank figure for make compatibility
    fig, ax = plt.subplots()
    outfile = FIG_DIR + 'meanr_scatter_set_figure.png'
    fig.savefig(outfile)

def filter_and_rescale(datablock, change_cas_corr, cutoff_bins):
    """
    Filter out low meanr and/or meanr values; returns (meanr_cas, meanr_cdp)
    """
    #get time-aligned nconc and meanr data 
    (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, change_cas_corr,
                                            cutoff_bins)
    (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, change_cas_corr,
                                            cutoff_bins)

    #filter out low values (have not done any sensitivity analysis for
    #these parameters)
    filter_inds = np.logical_and.reduce((
                    (nconc_cas > nconc_filter_val), \
                    (nconc_cdp > nconc_filter_val)))#, \
    #                (meanr_cas > meanr_filter_val), \
    #                (meanr_cdp > meanr_filter_val)))       

    #filter on LWC vals
    #booleanind = int(change_cas_corr) + int(cutoff_bins)*2
    #lwc_cas = datablock[:, high_bin_cas+booleanind]
    #lwc_cdp = datablock[:, high_bin_cdp+booleanind]
    #filter_inds = np.logical_and.reduce((
    #                (lwc_cas > lwc_filter_val), \
    #                (lwc_cdp > lwc_filter_val), \
    #                np.logical_not(np.isnan(meanr_cas)), \
    #                np.logical_not(np.isnan(meanr_cdp))))

    #filter spurious values and rescale to um
    meanr_cas = meanr_cas[filter_inds]*1.e6
    meanr_cdp = meanr_cdp[filter_inds]*1.e6
    
    return (meanr_cas, meanr_cdp)

def make_comparison_scatter(datablock_control, datablock_modified,
                    change_cas_corr, cutoff_bins, mod_label, suffix, date):
    """
    Make a scatter plot of CAS vs CDP meanr with two data sets, one the
    original "control" and the other corrected.
    """
    #get filtered and scaled data
    (meanr_cas_control, meanr_cdp_control) = \
        filter_and_rescale(datablock_control, False, False)
    (meanr_cas_modified, meanr_cdp_modified) = \
        filter_and_rescale(datablock_modified, change_cas_corr, cutoff_bins)
    
    #get linear fit params
    m_control, b_control, R_control, sig_control = \
        linregress(meanr_cas_control, meanr_cdp_control)
    m_modified, b_modified, R_modified, sig_modified = \
        linregress(meanr_cas_modified, meanr_cdp_modified)

    #get limits of the data for plotting purposes
    xlim = np.max(np.array( \
                    [np.max(meanr_cas_control), \
                    np.max(meanr_cdp_control), \
                    np.max(meanr_cas_modified), \
                    np.max(meanr_cdp_modified)]))
    ax_lims = np.array([0, xlim])

    #both correction schemes applied
    fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    ax.scatter(meanr_cas_control, meanr_cdp_control, \
                    c=colors['control'], \
                    label='Original data')
    ax.plot(ax_lims, np.add(b_control, m_control*ax_lims), \
                    c=colors['control'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_control, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_control**2, decimals=2))))
    ax.scatter(meanr_cas_modified, meanr_cdp_modified, \
                    c=colors['modified'], alpha=0.6, \
                    label=mod_label)
    ax.plot(ax_lims, np.add(b_modified, m_modified*ax_lims), \
                    c=colors['modified'], alpha=0.6, \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_modified, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_modified**2, decimals=2))))
    ax.plot(ax_lims, ax_lims, \
                    c='k', \
                    linestyle='dotted', \
                    linewidth=3, \
                    label='1:1')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(ax_lims)
    ax.set_ylim(ax_lims)
    ax.set_xlabel('CAS')
    ax.set_ylabel('CDP')
    handles, labels = ax.get_legend_handles_labels()

    #order set manually based on what matplotlib outputs by default
    order = [3, 0, 4, 1, 2]
    fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    fig.set_size_inches(21, 12)
    figtitle = 'Mean radius (um), Date: ' + date
    fig.suptitle(figtitle)
    outfile = FIG_DIR + versionstr + 'meanr_scatter_' + date + suffix + '.png'
    fig.savefig(outfile)
    plt.close()

if __name__ == "__main__":
    main()
