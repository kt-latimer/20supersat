"""
figure out them blobs!
"""
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, get_meanr_vs_t, \
                        get_nconc_vs_t, linregress, high_bin_cas, \
                        high_bin_cdp, pad_lwc_arrays

lwc_filter_val = 3.e-5
meanr_filter_val = 1.e-6
nconc_filter_val = 1.5e6

change_cas_corr = True
cutoff_bins = True

meanr_low_cas = 6
meanr_high_cas = 10
meanr_low_cdp = 12
meanr_high_cdp = 17

def main():
    """
    For dates with ``blobs'' in meanr CAS v CDP scatter plots under LWC
    filtering, get times of the points in the blobs. See `v6' or `v7'
    meanr_scatter_set figures at dates listed in the code.
    """

    dates_with_blob = ['20140906', '20140911', '20140916', \
                        '20140928', '20140930', '20141001']
    offsets = [3, 2, 3, 1, 3, 2]
    
    blob_pts = 0
    total_pts = 0

    for i, date in enumerate(dates_with_blob):
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

        #datablock with time offsets 
        #align all datasets along time
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(castime_offset), \
            np.around(cdptime)])
        datablock_offset = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock_offset):
            if sum(np.isnan(np.concatenate(((row[0:2], row[3:]))))) == 0:
                goodrows.append(j)
        datablock_offset = datablock_offset[goodrows, :]
        
        #sorry this is horrible code
        meanr_cas, meanr_cdp, lwc_cas, lwc_cdp, filtered_time = \
            filter_and_rescale(datablock_offset, change_cas_corr, cutoff_bins)

        blob_inds = np.logical_and.reduce((
                        (meanr_cas > meanr_low_cas), \
                        (meanr_cas < meanr_high_cas), \
                        (meanr_cdp > meanr_low_cdp), \
                        (meanr_cdp < meanr_high_cdp)))

        print(date, np.sum(blob_inds), np.shape(blob_inds)[0])
        print(filtered_time[blob_inds])
        blob_pts += np.sum(blob_inds)
        total_pts += np.shape(blob_inds)[0]
    
    print(blob_pts, total_pts, blob_pts/total_pts)

def filter_and_rescale(datablock, change_cas_corr, cutoff_bins):
    """
    Filter out low meanr and/or meanr values; returns (meanr_cas, meanr_cdp,
    lwc_cas, lwc_cdp, ptclnum_cas, ptclnum_cdp, filtered_time)
    """
    (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, change_cas_corr,
                                            cutoff_bins)
    (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, change_cas_corr,
                                            cutoff_bins)
    #filter out low values (have not done any sensitivity analysis for
    #these parameters)
    filter_inds = np.logical_and.reduce((
                    (nconc_cas > nconc_filter_val), \
                    (nconc_cdp > nconc_filter_val)))#, \

    ##filter on LWC vals
    booleanind = int(change_cas_corr) + int(cutoff_bins)*2
    lwc_cas = datablock[:, high_bin_cas+booleanind]
    lwc_cdp = datablock[:, high_bin_cdp+booleanind]
    #filter_inds = np.logical_and.reduce((
    #                (lwc_cas > lwc_filter_val), \
    #                (lwc_cdp > lwc_filter_val), \
    #                np.logical_not(np.isnan(meanr_cas)), \
    #                np.logical_not(np.isnan(meanr_cdp))))

    #filter spurious values and rescale to um
    meanr_cas = meanr_cas[filter_inds]*1.e6
    meanr_cdp = meanr_cdp[filter_inds]*1.e6
    lwc_cas = lwc_cas[filter_inds] 
    lwc_cdp = lwc_cdp[filter_inds]
    filtered_time = datablock[filter_inds, 0]

    return (meanr_cas, meanr_cdp, lwc_cas, lwc_cdp, filtered_time)

if __name__ == "__main__":
    main()
