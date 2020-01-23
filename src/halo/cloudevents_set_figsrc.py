"""
Create and save figure set cloudevents_set.
"""

import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_ind_bounds, match_multiple_arrays

matplotlib.rcParams.update({'font.size': 12})

casbinfile = DATA_DIR + 'CAS_bins.npy'
CAS_bins = np.load(casbinfile, allow_pickle=True).item()
cas_centr = (CAS_bins['upper'] + CAS_bins['lower'])/2.
cas_dr = CAS_bins['upper'] - CAS_bins['lower']
cas_nbins = len(cas_meanr)

cdpbinfile = DATA_DIR + 'CDP_bins.npy'
CDP_bins = np.load(cdpbinfile, allow_pickle=True).item()
cdp_centr = (CDP_bins['upper'] + CDP_bins['lower'])/2.
cdp_dr = CDP_bins['upper'] - CDP_bins['lower']
cdp_nbins = len(cdp_meanr)

#for plotting
colors = {'ADLR': '#777777', 'CAS': '#95B9E9', 'CDP': '#FC6A0C'}

#physical constants
Cp = 1005 #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
L = 2501000 #latent heat of evaporation of water (J/kg)
Mma=.02896 #Molecular weight of dry air (kg/mol)
Mmv=.01806 #Molecular weight of water vapour (kg/mol)
Rg = 8.317 #universal gas constant (J/(mol K))
Ra=Rg/Mma #Specific gas constant of dry air (J/(kg K))
Rv=Rg/Mmv #Specific gas constant of water vapour (J/(kg K))

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
        
        #entire lwc time sequence to plot in every figure for reference
        x3_adlr = adlrdata['data']['time'][adlrdata['data']['lwc_t_inds']]
        y3_adlr = adlrdata['data']['lwc']['11']
        x3_cas = casdata['data']['time'][casdata['data']['lwc_t_inds']]
        y3_cas = casdata['data']['lwc']['11']
        x3_cdp = cdpdata['data']['time'][cdpdata['data']['lwc_t_inds']]
        y3_cdp = cdpdata['data']['lwc']['11']
        
        #loop through all cloud events and make a figure for each
        t_tuples = 
            get_cloud_intervals(casdata['data']['lwc']['11'], \
            casdata['data']['lwc_t_inds'], \
            casdata['data']['time'], \
            cdpata['data']['time'])
        adlr_start_ind = 0
        cas_start_ind = 0
        cdp_start_ind = 0
        for i, t_tuple in enumerate(t_tuples):
            #get indices from datasets bounding the time of the cloud 
            #interval (as measured by that instrument). Add 7 second
            #buffer on either end to account for slight offsets between
            #instruments (from previous looks at these data this should
            #be sufficient in most cases - could miss edges at diffuse
            #and low-LWC cloud events
            tmin = t_tuple[0] - 7
            tmax = t_tuple[1] + 7
            adlr_ind_bounds = get_ind_bounds(adlrdata['data']['time'], \
                tmin, tmax, adlr_start_ind)
            adlr_start_ind = adlr_ind_bounds[1]
            cas_ind_bounds = get_ind_bounds(casdata['data']['time'], \
                tmin, tmax, cas_start_ind)
            cas_start_ind = cas_ind_bounds[1]
            cdp_ind_bounds = get_ind_bounds(cdpdata['data']['time'], \
                tmin, tmax, cdp_start_ind)
            cdp_start_ind = cdp_ind_bounds[1]
            #get indices of common time steps for eacdh dataset
            [adlrinds, casinds, cdpinds] = match_multiple_arrays(
                [adlr['data']['time'][adlr_ind_bounds[0]:adlr_ind_bounds[1]+1], \
                cas['data']['time'][cas_ind_bounds[0]:cas_ind_bounds[1]+1], \
                cdp['data']['time'][cdp_ind_bounds[0]:cdp_ind_bounds[1]+1]])
            datablock = get_datablock(adlrinds, casinds, cdpinds, \
                adlrdata, casdata, cdpdata)
            #remove rows with error values in any of the three
            goodrows = []
            for i, row in enumerate(datablock):
                if sum(np.isnan(row)) == 0:
                    goodrows.append(i)
            N = len(goodrows)
            Nerr = np.shape(datablock)[0] - N
            datablock = datablock[goodrows, :]
            
            #make and save figure for cloud event
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_grdspec(4, 5)
            
            #number concentration subplot
            ax1 = fig.add_subplot(gs[0:2, 0:4])
            x12_cas = cas_centr
            x12_cdp = cdp_centr
            (y1_cas, y1_cdp) = get_nconc_by_bin(datablock)
            ax1.plot(x12_cas, y1_cas, label='CAS', color=colors['CAS'])
            ax1.plot(x12_cdp, y1_cdp, label='CDP', color=colors['CDP'])
            ax1.ylabel('Average # Conc (m^-3)')
            
            #particle size pdf subplot
            ax2 = fig.add_subplot(gs[2:4, 0:4])
            y2_cas = y1_cas*cas_centr/cas_dr 
            y2_cdp = y1_cdp*cdp_centr/cdp_dr 
            ax2.plot(x12_cas, y2_cas, label='CAS', color=colors['CAS'])
            ax2.plot(x12_cdp, y2_cdp, label='CDP', color=colors['CDP'])
            ax2.xlabel('Central radius of bin (m)')
            ax2.ylabel('Particle radius PDF (n^-3)')

            #entire LWC timeseries subplot
            ax3 = fig.add_subplot(gs[0:2, 3:5])
            ax3.plot(x3_adlr, y3_adlr, label='ADLR', color=colors['ADLR'])
            ax3.plot(x3_cdp, y3_cdp, label='CDP', color=colors['CDP'])
            ax3.plot(x3_cas, y3_cas, label='CAS', color=colors['CAS'])
            tmid = np.mean([tmin, tmax])
            ax3.plot([tmid, tmid], ax3.get_ylim(), \
                    label='Cloud center', 'r:')
            ax3.xlabel('Time (s)')
            ax3.ylabel('LWC (g/g)')

            #ss vs time for just the cloud event
            ax4 = fig.add_subplot(gs[2:4, 3:5])
            x4 = datablock[:, 0]
            (y4_cas, y4_cdp) = get_ss_vs_T(datablock)
            ax4.plot(x4, y4_cas, label='CAS', color=colors['CAS'])
            ax4.plot(x4, y4_cdp, label='CDP', color=colors['CDP'])
            ax4.xlabel('Time (s)')
            ax4.ylabel('Supersaturation')
            
            figtitle = 'Date: ' + date + ' | Cloud event: ' + i + ' | N=' + str(N) + ' | Nerr=' + str(Nerr)
            fig.suptitle(figtitle, fontsize=14)
            outfile = FIG_DIR + 'cloudevents_' + date + '_' + str(int(tmin))
            fig.savefig(outfile)
            plt.close()

    #blank figure for make compatibility
    fig, ax = plt.subplots()
        
    outfile = FIG_DIR + 'cloudevents_set_figure.png'
    plt.savefig(outfile)

def get_cloud_intervals(caslwc, caslwctinds, cast, cdpt):
    """
    Given flight dataset, return list of tuples (t_min, t_max) where \
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
    return ttuples

def get_datablock(adlrinds, casinds, cdpinds, adlrdata, casdata, cdpdata):
    """
    Consolidate data for easier processing 
    Format of output array (order of columns): time, temperature, vertical \
    velocity, number conc for cas bins (__ cols), numer conc for cdp bins \
    (__ cols).
    """
    datablock = np.zeros(len(adlrinds), 3 + n_cas_bins, n_cdp_bins)
    datablock[:, 0] = np.around(adlrdata['data']['time'][adlrinds])
    datablock[:, 1] = adlrdata['data']['stat_temp'][adlrinds]
    datablock[:, 2] = adlrdata['data']['vert_vel'][adlrinds]
    
    for i in range(n_cas_bins):
        key = 'nconc' + str(i+5)
        datablock[:, i+3] = casdata['data'][key][casinds]
    
    for i in range(n_cdp_bins):
        key = 'nconc' + str(i+1)
        datablock[:, i+3+n_cas_bins] = cdpdata['data'][key][cdpinds]

    return datablock

def get_nconc_by_bin(datablock):
    """
    Returns (nconc_cas, nconc_cdp)
    """
    nconc_cas = []
    nconc_cdp = []
    for i in range(3, 3+n_cas_bins):
        nconc_cas.append(np.mean(datablock[:, i]))
    for i in range(3+n_cas_bins, 3+n_cas_bins+n_cdp_bins):
        nconc_cdp.append(np.mean(datablock[:, i]))
    return(nconc_cas, nconc_cdp)

def get_ss_vs_t(datablock):
    """
    Returns (ss_cas, ss_cdp).
    """
    T = datablock[:, 1] #temp
    u = datablock[:, 2] #vert vel
    one = np.ones(np.shape(T))
    A = g*(L*Ra/(Cp*R)*one/T - one)*1/Ra*one/T

    tot_nconc_cas = np.sum(datablock[:, 3:3+n_cas_bins], axis=?)
    meanr_cas = np.dot(datablock[:, 3:3+n_cas_bins], \
            np.transpose(cas_centr))/tot_nconc_cas
    ss_cas = 1/(4*np.pi*D)*A*u/(tot_nconc_cas*meanr_cas)

    tot_nconc_cdp = np.sum(datablock[:, 3:3+n_cdp_bins], axis=?)
    meanr_cdp = np.dot(datablock[:, 3:3+n_cdp_bins], \
            np.transpose(cdp_centr))/tot_nconc_cdp
    ss_cdp = 1/(4*np.pi*D)*A*u/(tot_nconc_cdp*meanr_cdp)

    return (ss_cas, ss_cdp)

if __name__ == "__main__":
    main()
