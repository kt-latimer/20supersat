"""
Create and save figure set cloudevents_set.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 22})

def main():
    """
    for all dates listed (manually in code for now), get cloud events \
    and create/save a figure containing:
    -average number concentration by mean bin radius (=<nconc>_i) for \
    both CAS and CDP
    -particle size pdf by mean bin radius (=<r>_i*<nconc>_i/(dr)_i) for \
    both CAS and CDP
    -inset of plot from comparelwc set (using cutoff_bins = change_cas_corr \
    = True) with arrow indicating peak center
    -supersaturation vs time for both CAS and CDP
    """

    dates = ['20140909', '20141001']
    
    for date in dates:
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()
        
        #loop through all cloud events and make a figure for each
        (ttuples, casindtuples, cdpindtuples) = 
            get_cloud_intervals(casdata['data']['lwc']['11'], \
            casdata['data']['lwc_t_inds'], \
            casdata['data']['time'], \
            cdpata['data']['time'])
        for i, ttuple in enumerate(ttuples):
            casindtuple = casindtuples[i]
            cdpindtuple = cdpindtuples[i]
            (casnconc, casradpdf) = \
                get_nconc_and_radpdf_vs_bins(casindtuple, casdata)
            (casss, casss_t) = get_ss_vs_t(casindtuple, casdata, adlrdata)
            (cdpnconc, cdpradpdf) = \
                get_nconc_and_radpdf_vs_bins(cdpindtuple, cdpdata)
            (cdpss, cdpss_t) = get_ss_vs_t(cdpindtuple, cdpdata, adlrdata)
            make_cloud_event_figure(date, ttuple, casnconc, cdpnconc, \
                casradpdf, cdpradpdf, casss, cdpss, casss_t, cdpss_t)
    
    #blank figure for make compatibility
    fig, ax = plt.subplots()
        
    outfile = FIG_DIR + 'cloudevents_set_figure.png'
    plt.savefig(outfile)

def get_cloud_intervals(caslwc, caslwctinds, cast, cdpt):
    """
    Given flight dataset, return list of tuples (t_min, t_max)* where \
    t_min is when the cloud event starts and t_max is where it ends. \
    I am currently using the CAS data to find these after playing around \
    with peak selection; algorithm looks for: 
    -LWC > 1.e-6 g/g
    -Delta_t between cloud events > 10 s
    -Half-max of LWC peak > 7.e-6 g/g
    These parameters are somewhat arbitrary (also below previous 1.e5 \
    LWC cutoff for 'cloud' designation in this project), but gave peak \
    selection that roughly matched what I would pick by eye. Not perfect \
    but does the job.

    *EDIT: for convenience in other code, return:
    ([<tuples (tmin, tmax)>], \
    [<tuples (imin_cas, imax_cas)>], \
    [<tuples (imin_cdp, imax_cdp)>])
    """
    #get indices of cas clouds
    cas_cloud_clusters = []
    current_cloud = []
    in_cloud = False
    for i, val in enumerate(caslwc):
        if val > 1.e-6:
            if in_cloud:
                current_cloud.append(i)
            else:
                in_cloud = True
                current_cloud.append(i)
        else:
            if in_cloud:
                cas_cloud_clusters.append(current_cloud)
                current_cloud = []
                in_cloud = False
    
    #group clusters separated by less than ten seconds
    big_cas_clusters = []
    big_cluster = cas_cloud_clusters[0]
    for i, cluster in enumerate(cas_cloud_clusters[:-1]):
        if cast[cas_cloud_clusters[i+1][0]] - cast[cluster[-1]] < 10:
            big_cluster += cas_cloud_clusters[i+1]
        else:
            big_cas_clusters.append(big_cluster)
            big_cluster = cas_cloud_clusters[i+1]
    #add last one
    big_cas_clusters.append(big_cluster)
    
    ttuples = []
    casindtuples = []
    cdpindtuples = []
    startind = 0

    #pick peaks with sufficient half-max
    for thing in big_cas_clusters:
        if np.max(caslwc[thing])/2. > 0.7e-5:
            imin = big_cluster[0]
            imax = big_cluster[-1]
            ttuples.append((cast[imin], cast[imax]))
            casindtuples.append((imin, imax))
            cdpinds = get_ind_bounds(cdpt, \
                cast[imin], cast[imax], startind)
            cdpindtuples.append(cdpinds)
            startind = cdpinds[1]
    
    return(ttuples, casindtuples, cdpindtuples)

def get_ind_bounds(arr, minval, maxval, startind):
    i = startind
    while arr[i] < minval:
        i += 1
    imin = i
    while arr[i] < maxval:
        i += 1
    imax = i
    return(imin, imax)

def get_nconc_and_radpdf_vs_bins():
    """
    given setname (CAS or CDP), dataset, and time interval, return array \
    of average number concentrations and radial PDF for each bin over \
    that interval (using 3um bin cutoff)
    """
    return(nconc, radpdf)

def get_ss_vs_t():
    """
    given 
    """
    print('bleh')

def make_cloud_event_figure(date, interval, casnconc, cdpnconc, casradpdf, \
        cdpradpdf, casss, cdpss, casss_t, cdpss_t):
    """
    make and save complete figure for cloud event - figure name is in the \
    format `cloudevents_YYYYMMDD_tmin.png' where tmin is the (rounded) \
    beginning of the cloud event as measured by CAS instrument in UTC secs
    """
    
    outfile = FIG_DIR + 'cloudevents_' + date + '_' \
            + str(int(np.round(tint[0]))) + '.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
