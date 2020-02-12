"""
Create and save figure cascorr_nconc_set.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_nconc_vs_t, \
                        get_meanr_vs_t, linregress

#for plotting
colors = {'original': '#777777', 'modified': '#FC6A0C'}
prefix = 'v4'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

nconc_filter_val = 10.e6
meanr_filter_val = 1.e-6

def main():
    """
    the main routine.
    """
    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
            '20140918', '20140921', '20140927', '20140928', \
            '20140930', '20141001']
    offsets = [3, 2, 2, 3, 3, 2, 1, 3, 1, 3, 2]
    
    for i, date in enumerate(dates):
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        casdata['data']['time'] = \
            np.array([t - offsets[i] for t in casdata['data']['time']])
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

        #get time-aligned nconc and meanr data for comparison of correction
        #schemes
        (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, False, False)
        (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, False, False)
        filter_inds = np.logical_and.reduce((
                        (nconc_cas > nconc_filter_val), \
                        (nconc_cdp > nconc_filter_val), \
                        (meanr_cas > meanr_filter_val), \
                        (meanr_cdp > meanr_filter_val)))

        (nconc_corr_cas, nconc_corr_cdp) = \
            get_nconc_vs_t(datablock, True, False)
        (meanr_corr_cas, meanr_corr_cdp) = \
            get_meanr_vs_t(datablock, True, False)
        filter_inds_corr = np.logical_and.reduce((
                        (nconc_corr_cas > nconc_filter_val), \
                        (nconc_corr_cdp > nconc_filter_val), \
                        (meanr_corr_cas > meanr_filter_val), \
                        (meanr_corr_cdp > meanr_filter_val)))

        (nconc_cutoff_cas, nconc_cutoff_cdp) = \
            get_nconc_vs_t(datablock, False, True)
        (meanr_cutoff_cas, meanr_cutoff_cdp) = \
            get_meanr_vs_t(datablock, False, True)
        filter_inds_cutoff = np.logical_and.reduce((
                        (nconc_cutoff_cas > nconc_filter_val), \
                        (nconc_cutoff_cdp > nconc_filter_val), \
                        (meanr_cutoff_cas > meanr_filter_val), \
                        (meanr_cutoff_cdp > meanr_filter_val)))
        
        (nconc_both_cas, nconc_both_cdp) = \
            get_nconc_vs_t(datablock, True, True)
        (meanr_both_cas, meanr_both_cdp) = \
            get_meanr_vs_t(datablock, True, True)
        filter_inds_both = np.logical_and.reduce((
                        (nconc_both_cas > nconc_filter_val), \
                        (nconc_both_cdp > nconc_filter_val), \
                        (meanr_both_cas > meanr_filter_val), \
                        (meanr_both_cdp > meanr_filter_val)))
        
        #apply num conc and mean radius filters and change to units of ccm
        nconc_cas = nconc_cas[filter_inds]*1.e-6
        nconc_cdp = nconc_cdp[filter_inds]*1.e-6

        nconc_corr_cas = nconc_corr_cas[filter_inds]*1.e-6
        nconc_corr_cdp = nconc_corr_cdp[filter_inds]*1.e-6

        nconc_cutoff_cas = nconc_cutoff_cas[filter_inds]*1.e-6
        nconc_cutoff_cdp = nconc_cutoff_cdp[filter_inds]*1.e-6

        nconc_both_cas = nconc_both_cas[filter_inds]*1.e-6
        nconc_both_cdp = nconc_both_cdp[filter_inds]*1.e-6

        #get linear regression params
        m, b, R, sig = linregress(nconc_cas, nconc_cdp)
        m_corr, b_corr, R_corr, sig_corr = \
            linregress(nconc_corr_cas, nconc_corr_cdp)
        m_cutoff, b_cutoff, R_cutoff, sig_cutoff = \
            linregress(nconc_cutoff_cas, nconc_cutoff_cdp)
        m_both, b_both, R_both, sig_both = \
            linregress(nconc_both_cas, nconc_both_cdp)

        #get enpoints for plotting lines
        xlim_corr = np.max(np.array( \
                        [np.max(nconc_cas), \
                        np.max(nconc_cdp), \
                        np.max(nconc_corr_cas), \
                        np.max(nconc_corr_cdp)]))
        ax_lims_corr = np.array([0, xlim_corr])
        
        xlim_cutoff = np.max(np.array( \
                        [np.max(nconc_cas), \
                        np.max(nconc_cdp), \
                        np.max(nconc_cutoff_cas), \
                        np.max(nconc_cutoff_cdp)]))
        ax_lims_cutoff = np.array([0, xlim_cutoff])

        xlim_both = np.max(np.array( \
                        [np.max(nconc_cas), \
                        np.max(nconc_cdp), \
                        np.max(nconc_both_cas), \
                        np.max(nconc_both_cdp)]))
        ax_lims_both = np.array([0, xlim_both])

        #num concs with cas corrected by xi (note nconc_corr_cdp
        #is the same as nconc_cdp but giving them different names
        #for clarity)
        corr_fig, corr_ax = plt.subplots()
        corr_ax.scatter(nconc_cas, nconc_cdp, \
                        c=colors['original'], \
                        label='Different correction factors')
        corr_ax.plot(ax_lims_corr, np.add(b, m*ax_lims_corr), \
                        c=colors['original'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        corr_ax.scatter(nconc_corr_cas, nconc_corr_cdp, \
                        c=colors['modified'], alpha=0.6, \
                        label='Same correction factors')
        corr_ax.plot(ax_lims_corr, np.add(b_corr, m_corr*ax_lims_corr), \
                        c=colors['modified'], alpha=0.6, \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m_corr, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R_corr**2, decimals=2))))
        corr_ax.plot(ax_lims_corr, ax_lims_corr, \
                        c='k', \
                        linestyle='dotted', \
                        linewidth=3, \
                        label='1:1')
        corr_ax.set_aspect('equal', 'box')
        corr_ax.set_xlim(ax_lims_corr)
        corr_ax.set_ylim(ax_lims_corr)
        corr_ax.set_xlabel('CAS')
        corr_ax.set_ylabel('CDP')
        handles, labels = corr_ax.get_legend_handles_labels()
        #order set manually based on what matplotlib outputs by default
        order = [3, 0, 4, 1, 2]
        corr_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        corr_fig.set_size_inches(21, 12)
        corr_figtitle = ' Number concentration (cm^-3), Date: ' + date
        corr_fig.suptitle(corr_figtitle)
        outfile = FIG_DIR + prefix + 'nconc_scatter_cascorr_' + date + '.png'
        corr_fig.savefig(outfile)
        
        #num concs with 3um bin cutoff
        cutoff_fig, cutoff_ax = plt.subplots()
        cutoff_fig, cutoff_ax = plt.subplots()
        cutoff_ax.scatter(nconc_cas, nconc_cdp, \
                        c=colors['original'], \
                        label='No lower bin cutoff')
        cutoff_ax.plot(ax_lims_cutoff, np.add(b, m*ax_lims_cutoff), \
                        c=colors['original'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        cutoff_ax.scatter(nconc_cutoff_cas, nconc_cutoff_cdp, \
                        c=colors['modified'], alpha=0.6, \
                        label='With lower bin cutoff')
        cutoff_ax.plot(ax_lims_cutoff, np.add(b_cutoff, m_cutoff*ax_lims_cutoff), \
                        c=colors['modified'], alpha=0.6, \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m_cutoff, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R_cutoff**2, decimals=2))))
        cutoff_ax.plot(ax_lims_cutoff, ax_lims_cutoff, \
                        c='k', \
                        linestyle='dotted', \
                        linewidth=3, \
                        label='1:1')
        cutoff_ax.set_aspect('equal', 'box')
        cutoff_ax.set_xlim(ax_lims_cutoff)
        cutoff_ax.set_ylim(ax_lims_cutoff)
        cutoff_ax.set_xlabel('CAS')
        cutoff_ax.set_ylabel('CDP')
        handles, labels = cutoff_ax.get_legend_handles_labels()
        #order set manually based on what matplotlib outputs by default
        order = [3, 0, 4, 1, 2]
        cutoff_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        cutoff_fig.set_size_inches(21, 12)
        cutoff_figtitle = ' Number concentration (cm^-3), Date: ' + date
        cutoff_fig.suptitle(cutoff_figtitle)
        outfile = FIG_DIR + prefix + 'nconc_scatter_cutoff_' + date + '.png'
        cutoff_fig.savefig(outfile)
   
        #both correction schemes applied
        both_fig, both_ax = plt.subplots()
        both_fig, both_ax = plt.subplots()
        both_ax.scatter(nconc_cas, nconc_cdp, \
                        c=colors['original'], \
                        label='No modifications applied')
        both_ax.plot(ax_lims_both, np.add(b, m*ax_lims_both), \
                        c=colors['original'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        both_ax.scatter(nconc_both_cas, nconc_both_cdp, \
                        c=colors['modified'], alpha=0.6, \
                        label='Apply xi factor to CAS, cutoff bins at 3um')
        both_ax.plot(ax_lims_both, np.add(b_both, m_both*ax_lims_both), \
                        c=colors['modified'], alpha=0.6, \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m_both, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R_both**2, decimals=2))))
        both_ax.plot(ax_lims_both, ax_lims_both, \
                        c='k', \
                        linestyle='dotted', \
                        linewidth=3, \
                        label='1:1')
        both_ax.set_aspect('equal', 'box')
        both_ax.set_xlim(ax_lims_both)
        both_ax.set_ylim(ax_lims_both)
        both_ax.set_xlabel('CAS')
        both_ax.set_ylabel('CDP')
        handles, labels = both_ax.get_legend_handles_labels()
        #order set manually based on what matplotlib outputs by default
        order = [3, 0, 4, 1, 2]
        both_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
        both_fig.set_size_inches(21, 12)
        both_figtitle = ' Number concentration (cm^-3), Date: ' + date
        both_fig.suptitle(both_figtitle)
        outfile = FIG_DIR + prefix + 'nconc_scatter_both_' + date + '.png'
        both_fig.savefig(outfile)
   
        #collect aggregate data
        if i == 0:
            all_dates_nconc_cas = nconc_cas 
            all_dates_nconc_cdp = nconc_cdp 
            
            all_dates_nconc_corr_cas = nconc_corr_cas 
            all_dates_nconc_corr_cdp = nconc_corr_cdp 
            
            all_dates_nconc_cutoff_cas = nconc_cutoff_cas 
            all_dates_nconc_cutoff_cdp = nconc_cutoff_cdp 
            
            all_dates_nconc_both_cas = nconc_both_cas 
            all_dates_nconc_both_cdp = nconc_both_cdp 
        else:
            all_dates_nconc_cas = \
                np.concatenate((all_dates_nconc_cas, nconc_cas)) 
            all_dates_nconc_cdp = \
                np.concatenate((all_dates_nconc_cdp, nconc_cdp))
            
            all_dates_nconc_corr_cas = \
                np.concatenate((all_dates_nconc_corr_cas, \
                nconc_corr_cas))
            all_dates_nconc_corr_cdp = \
                np.concatenate((all_dates_nconc_corr_cdp, \
                nconc_corr_cdp))

            all_dates_nconc_cutoff_cas = \
                np.concatenate((all_dates_nconc_cutoff_cas, \
                nconc_cutoff_cas)) 
            all_dates_nconc_cutoff_cdp = \
                np.concatenate((all_dates_nconc_cutoff_cdp, \
                nconc_cutoff_cdp))

            all_dates_nconc_both_cas = \
                np.concatenate((all_dates_nconc_both_cas, \
                nconc_both_cas)) 
            all_dates_nconc_both_cdp = \
                np.concatenate((all_dates_nconc_both_cdp, \
                nconc_both_cdp))

    #aggregate all dates to plot
    #get linear regression params
    m, b, R, sig = linregress(all_dates_nconc_cas, all_dates_nconc_cdp)
    m_corr, b_corr, R_corr, sig_corr = \
        linregress(all_dates_nconc_corr_cas, all_dates_nconc_corr_cdp)
    m_cutoff, b_cutoff, R_cutoff, sig_cutoff = \
        linregress(all_dates_nconc_cutoff_cas, all_dates_nconc_cutoff_cdp)
    m_both, b_both, R_both, sig_both = \
        linregress(all_dates_nconc_both_cas, all_dates_nconc_both_cdp)

    #get enpoints for plotting lines
    xlim_corr = np.max(np.array( \
                    [np.max(all_dates_nconc_cas), \
                    np.max(all_dates_nconc_cdp), \
                    np.max(all_dates_nconc_corr_cas), \
                    np.max(all_dates_nconc_corr_cdp)]))
    ax_lims_corr = np.array([0, xlim_corr])
    
    xlim_cutoff = np.max(np.array( \
                    [np.max(all_dates_nconc_cas), \
                    np.max(all_dates_nconc_cdp), \
                    np.max(all_dates_nconc_cutoff_cas), \
                    np.max(all_dates_nconc_cutoff_cdp)]))
    ax_lims_cutoff = np.array([0, xlim_cutoff])

    xlim_both = np.max(np.array( \
                    [np.max(all_dates_nconc_cas), \
                    np.max(all_dates_nconc_cdp), \
                    np.max(all_dates_nconc_both_cas), \
                    np.max(all_dates_nconc_both_cdp)]))
    ax_lims_both = np.array([0, xlim_both])

    #num concs with cas corrected by xi (note all_dates_nconc_corr_cdp
    #is the same as all_dates_nconc_cdp but giving them different names
    #for clarity)
    corr_fig, corr_ax = plt.subplots()
    corr_ax.scatter(all_dates_nconc_cas, all_dates_nconc_cdp, \
                    c=colors['original'], \
                    label='Different correction factors')
    corr_ax.plot(ax_lims_corr, np.add(b, m*ax_lims_corr), \
                    c=colors['original'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R**2, decimals=2))))
    corr_ax.scatter(all_dates_nconc_corr_cas, all_dates_nconc_corr_cdp, \
                    c=colors['modified'], alpha=0.6, \
                    label='Same correction factors')
    corr_ax.plot(ax_lims_corr, np.add(b_corr, m_corr*ax_lims_corr), \
                    c=colors['modified'], alpha=0.6, \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_corr, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_corr**2, decimals=2))))
    corr_ax.plot(ax_lims_corr, ax_lims_corr, \
                    c='k', \
                    linestyle='dotted', \
                    linewidth=3, \
                    label='1:1')
    corr_ax.set_aspect('equal', 'box')
    corr_ax.set_xlim(ax_lims_corr)
    corr_ax.set_ylim(ax_lims_corr)
    corr_ax.set_xlabel('CAS')
    corr_ax.set_ylabel('CDP')
    handles, labels = corr_ax.get_legend_handles_labels()
    #order set manually based on what matplotlib outputs by default
    order = [3, 0, 4, 1, 2]
    corr_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    corr_fig.set_size_inches(21, 12)
    corr_figtitle = ' Number concentration (cm^-3), all dates'
    corr_fig.suptitle(corr_figtitle)
    outfile = FIG_DIR + prefix + 'nconc_scatter_cascorr_all.png'
    corr_fig.savefig(outfile)
    
    #num concs with 3um bin cutoff
    cutoff_fig, cutoff_ax = plt.subplots()
    cutoff_fig, cutoff_ax = plt.subplots()
    cutoff_ax.scatter(all_dates_nconc_cas, all_dates_nconc_cdp, \
                    c=colors['original'], \
                    label='No lower bin cutoff')
    cutoff_ax.plot(ax_lims_cutoff, np.add(b, m*ax_lims_cutoff), \
                    c=colors['original'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R**2, decimals=2))))
    cutoff_ax.scatter(all_dates_nconc_cutoff_cas, all_dates_nconc_cutoff_cdp, \
                    c=colors['modified'], alpha=0.6, \
                    label='With lower bin cutoff')
    cutoff_ax.plot(ax_lims_cutoff, np.add(b_cutoff, m_cutoff*ax_lims_cutoff), \
                    c=colors['modified'], alpha=0.6, \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_cutoff, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_cutoff**2, decimals=2))))
    cutoff_ax.plot(ax_lims_cutoff, ax_lims_cutoff, \
                    c='k', \
                    linestyle='dotted', \
                    linewidth=3, \
                    label='1:1')
    cutoff_ax.set_aspect('equal', 'box')
    cutoff_ax.set_xlim(ax_lims_cutoff)
    cutoff_ax.set_ylim(ax_lims_cutoff)
    cutoff_ax.set_xlabel('CAS')
    cutoff_ax.set_ylabel('CDP')
    handles, labels = cutoff_ax.get_legend_handles_labels()
    #order set manually based on what matplotlib outputs by default
    order = [3, 0, 4, 1, 2]
    cutoff_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    cutoff_fig.set_size_inches(21, 12)
    cutoff_figtitle = 'Number concentration (cm^-3), all dates'
    cutoff_fig.suptitle(cutoff_figtitle)
    outfile = FIG_DIR + prefix + 'nconc_scatter_cutoff_all.png'
    cutoff_fig.savefig(outfile)

    #both correction schemes applied
    cutoff_fig, cutoff_ax = plt.subplots()
    cutoff_fig, cutoff_ax = plt.subplots()
    cutoff_ax.scatter(all_dates_nconc_cas, all_dates_nconc_cdp, \
                    c=colors['original'], \
                    label='No modifications applied')
    cutoff_ax.plot(ax_lims_cutoff, np.add(b, m*ax_lims_cutoff), \
                    c=colors['original'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R**2, decimals=2))))
    cutoff_ax.scatter(all_dates_nconc_cutoff_cas, all_dates_nconc_cutoff_cdp, \
                    c=colors['modified'], alpha=0.6, \
                    label='Apply xi factor to CAS, cutoff bins at 3um')
    cutoff_ax.plot(ax_lims_cutoff, np.add(b_cutoff, m_cutoff*ax_lims_cutoff), \
                    c=colors['modified'], alpha=0.6, \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_cutoff, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_cutoff**2, decimals=2))))
    cutoff_ax.plot(ax_lims_cutoff, ax_lims_cutoff, \
                    c='k', \
                    linestyle='dotted', \
                    linewidth=3, \
                    label='1:1')
    cutoff_ax.set_aspect('equal', 'box')
    cutoff_ax.set_xlim(ax_lims_cutoff)
    cutoff_ax.set_ylim(ax_lims_cutoff)
    cutoff_ax.set_xlabel('CAS')
    cutoff_ax.set_ylabel('CDP')
    handles, labels = cutoff_ax.get_legend_handles_labels()
    #order set manually based on what matplotlib outputs by default
    order = [3, 0, 4, 1, 2]
    cutoff_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    cutoff_fig.set_size_inches(21, 12)
    cutoff_figtitle = 'Number concentration (cm^-3), all dates'
    cutoff_fig.suptitle(cutoff_figtitle)
    outfile = FIG_DIR + prefix + 'nconc_scatter_cutoff_all.png'
    cutoff_fig.savefig(outfile)

    #both correction schemes applied
    both_fig, both_ax = plt.subplots()
    both_fig, both_ax = plt.subplots()
    both_ax.scatter(all_dates_nconc_cas, all_dates_nconc_cdp, \
                    c=colors['original'], \
                    label='No modifications applied')
    both_ax.plot(ax_lims_both, np.add(b, m*ax_lims_both), \
                    c=colors['original'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R**2, decimals=2))))
    both_ax.scatter(all_dates_nconc_both_cas, all_dates_nconc_both_cdp, \
                    c=colors['modified'], alpha=0.6, \
                    label='Apply xi factor to CAS, cutoff bins at 3um')
    both_ax.plot(ax_lims_both, np.add(b_both, m_both*ax_lims_both), \
                    c=colors['modified'], alpha=0.6, \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m_both, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R_both**2, decimals=2))))
    both_ax.plot(ax_lims_both, ax_lims_both, \
                    c='k', \
                    linestyle='dotted', \
                    linewidth=3, \
                    label='1:1')
    both_ax.set_aspect('equal', 'box')
    both_ax.set_xlim(ax_lims_both)
    both_ax.set_ylim(ax_lims_both)
    both_ax.set_xlabel('CAS')
    both_ax.set_ylabel('CDP')
    handles, labels = both_ax.get_legend_handles_labels()
    #order set manually based on what matplotlib outputs by default
    order = [3, 0, 4, 1, 2]
    both_fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    both_fig.set_size_inches(21, 12)
    both_figtitle = ' Number concentration (cm^-3), Date: ' + date
    both_fig.suptitle(both_figtitle)
    outfile = FIG_DIR + prefix + 'all_dates_nconc_scatter_both.png'
    both_fig.savefig(outfile)

    #blank figure for make compatibility
    fig, ax = plt.subplots()
    outfile = FIG_DIR + prefix + 'nconc_scatter_set_figure.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
