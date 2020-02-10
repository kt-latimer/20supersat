"""
Create and save figure set cloudevents_set.
"""

import matplotlib
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
			match_multiple_arrays, get_nconc_vs_t, \
			get_meanr_vs_t,  nbins_cas, nbins_cdp, \
			centr_cas, centr_cdp, dr_cas, dr_cdp, \
			low_bin_cas, high_bin_cas, low_bin_cdp, \
			high_bin_cdp

matplotlib.rcParams.update({'font.size': 14})

#boolean params; change manually in code for now
cutoff_bins = True
change_cas_corr = True
booleankey = str(int(cutoff_bins)) + str(int(change_cas_corr))

#low_bin_cas = 4 
#high_bin_cas = low_bin_cas + nbins_cas
#low_bin_cdp = high_bin_cas 
#high_bin_cdp = low_bin_cdp + nbins_cdp
#print(low_bin_cas)

#for plotting
colors = {'ADLR': '#777777', 'CAS': '#95B9E9', 'CDP': '#FC6A0C', 'c1': '#BA3F00', 'c2': '#095793'}

#physical constants
Cp = 1005 #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
L = 2501000 #latent heat of evaporation of water (J/kg)
Mma=.02896 #Molecular weight of dry air (kg/mol)
Mmv=.01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
Ra=R/Mma #Specific gas constant of dry air (J/(kg K))
Rv=R/Mmv #Specific gas constant of water vapour (J/(kg K))

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

    dates = ['20140909', '20140911', '20141001']
    offsets = [2, 2, 3]

    for m, date in enumerate(dates):
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
	#time offset (estimated by eye so not super replicable atm)
        casdata['data']['time'] = np.array([t - offsets[m] for t in casdata['data']['time']])
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()
        
        #entire lwc time sequence to plot in every figure for reference
        x3_adlr = adlrdata['data']['time']
        y3_adlr = adlrdata['data']['lwc']
        x3_cas = casdata['data']['time'][casdata['data']['lwc_t_inds']]
        y3_cas = casdata['data']['lwc'][booleankey]
        x3_cdp = cdpdata['data']['time'][cdpdata['data']['lwc_t_inds']]
        y3_cdp = cdpdata['data']['lwc'][booleankey]
        
        #loop through all cloud events and make a figure for each
        t_tuples = \
            get_cloud_intervals(casdata['data']['lwc'][booleankey], \
            casdata['data']['time'][casdata['data']['lwc_t_inds']])
        adlr_start_ind = 0
        cas_start_ind = 0
        cdp_start_ind = 0

        for j, t_tuple in enumerate(t_tuples):
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
                [np.around(adlrdata['data']['time'][adlr_ind_bounds[0]:adlr_ind_bounds[1]+1]), \
                np.around(casdata['data']['time'][cas_ind_bounds[0]:cas_ind_bounds[1]+1]), \
                np.around(cdpdata['data']['time'][cdp_ind_bounds[0]:cdp_ind_bounds[1]+1])])
            adlrinds = [ind + adlr_ind_bounds[0] for ind in adlrinds]
            casinds = [ind + cas_ind_bounds[0] for ind in casinds]
            cdpinds = [ind + cdp_ind_bounds[0] for ind in cdpinds]
            datablock = get_datablock(adlrinds, casinds, cdpinds, \
                adlrdata, casdata, cdpdata, change_cas_corr)
            
            #remove rows with error values in any of the three
            goodrows = []
            for i, row in enumerate(datablock):
                if sum(np.isnan(np.concatenate((row[0:2], row[3:])))) == 0:
                    goodrows.append(i)
            N = len(goodrows)
            Nerr = np.shape(datablock)[0] - N
            datablock = datablock[goodrows, :]
            
            #make and save figure for cloud event
            fig = plt.figure(constrained_layout=True)
            fig.set_size_inches(21, 12)
            gs = fig.add_gridspec(4, 5)
            
            if cutoff_bins:
                cas_offset = 3
                cdp_offset = 2
            else:
                cas_offset = 0
                cdp_offset = 0

            #number concentration pdf subplot
            ax1 = fig.add_subplot(gs[0:2, 0:2])
            x12_cas = 1.e6*centr_cas[nbins_cas-(high_bin_cas-(low_bin_cas+cas_offset)):nbins_cas]
            x12_cdp = 1.e6*centr_cdp[nbins_cdp-(high_bin_cdp-(low_bin_cdp+cdp_offset)):nbins_cdp]
            (y1_cas, y1_cdp) = get_nconc_by_bin(datablock, change_cas_corr, cutoff_bins) 
            y1_cas = y1_cas/dr_cas[nbins_cas-(high_bin_cas-(low_bin_cas+cas_offset)):nbins_cas]
            y1_cdp = y1_cdp/dr_cdp[nbins_cdp-(high_bin_cdp-(low_bin_cdp+cdp_offset)):nbins_cdp]
            ax1.plot(x12_cas, y1_cas, label='CAS', marker='o', color=colors['CAS'])
            ax1.plot(x12_cdp, y1_cdp, label='CDP', marker='o', color=colors['CDP'])
            ax1.set_ylabel('# Conc PDF (m^-4)')
            ax1.ticklabel_format(axis='both', style='sci')
            ax1.legend()

            #particle size pdf subplot
            ax2 = fig.add_subplot(gs[2:4, 0:2])
            y2_cas = y1_cas*centr_cas[nbins_cas-(high_bin_cas-(low_bin_cas+cas_offset)):nbins_cas] 
            y2_cdp = y1_cdp*centr_cdp[nbins_cdp-(high_bin_cdp-(low_bin_cdp+cdp_offset)):nbins_cdp] 
            ax2.plot(x12_cas, y2_cas, label='CAS', marker='o', color=colors['CAS'])
            ax2.plot(x12_cdp, y2_cdp, label='CDP', marker='o', color=colors['CDP'])
            ax2.set_xlabel('Central radius of bin (um)')
            ax2.set_ylabel('Particle radius PDF (m^-3)')
            ax2.ticklabel_format(axis='both', style='sci')

	    #entire LWC timeseries subplot
            ax3 = fig.add_subplot(gs[0:2, 2:5])
            ax3.plot(x3_adlr, y3_adlr, label='ADLR', color=colors['ADLR'])
            ax3.plot(x3_cdp, y3_cdp, label='CDP', color=colors['CDP'])
            ax3.plot(x3_cas, y3_cas, label='CAS', color=colors['CAS'])
            tmid = np.mean([tmin, tmax])
            ax3.plot([tmid, tmid], ax3.get_ylim(), \
                'r:', label='Cloud center')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('LWC (g/g)')
            ax3.ticklabel_format(axis='x', style='sci')
            ax3.legend()

            #cloud drop data just for the cloud event
            ax4 = fig.add_subplot(gs[2:3, 2:5])
            x456 = datablock[:, 0]
            (y4_cas, y4_cdp) = get_nconc_vs_t(datablock, change_cas_corr, cutoff_bins)
            (y5_cas, y5_cdp) = get_meanr_vs_t(datablock, change_cas_corr, cutoff_bins)
            if N != 0:
                y5_cas = 1.e6*y5_cas
                y5_cdp = 1.e6*y5_cdp
            ax4.plot(x456, y4_cas, label='cas nconc', marker='o', color=colors['CAS'])
            ax4.plot(x456, y4_cdp, label='cdp nconc', marker='o', color=colors['CDP'])
            ax4.set_ylabel('nconc (m^-3)')
            ax5 = ax4.twinx()
            ax5.plot(x456, y5_cas, label='cas meanr', marker='o', color=colors['c2'])
            ax5.plot(x456, y5_cdp, label='cdp meanr', marker='o', color=colors['c1'])
            ax5.set_ylabel('meanr (um)')
            lines4, labels4 = ax4.get_legend_handles_labels() 
            lines5, labels5 = ax5.get_legend_handles_labels() 
            ax5.legend(lines4 + lines5, labels4 + labels5, loc=0)
            
            ##env vars vs time just for the cloud event
            #ax4 = fig.add_subplot(gs[2:3, 2:5])
            #x456 = datablock[:, 0]
            #y4 = get_vert_wind_vel_vs_t(datablock)
            #y5 = get_A_vs_t(datablock)
            #ax4.plot(x456, y4, label='w', marker='o', color=colors['c1'])
            #ax4.set_ylabel('w (m/s)')
            #ax5 = ax4.twinx()
            #ax5.plot(x456, y5, label='A', marker='o', color=colors['c2'])
            #ax5.set_ylabel('A')
            
            #ss vs time for just the cloud event
            ax6 = fig.add_subplot(gs[3:4, 2:5])
            (y6_cas, y6_cdp) = get_ss_vs_t(datablock, change_cas_corr, cutoff_bins)
            ax6.plot(x456, y6_cas, label='CAS', marker='o', color=colors['CAS'])
            ax6.plot(x456, y6_cdp, label='CDP', marker='o', color=colors['CDP'])
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Supersaturation')
            ax6.ticklabel_format(axis='x', style='sci')
            
            figtitle = 'Date: ' + date + ' | Cloud event: ' + str(j) \
                    + ' | N=' + str(N) + ' | Nerr=' + str(Nerr)
            fig.suptitle(figtitle, fontsize=14)
            outfile = FIG_DIR + 'v3cloudevents_' + date + '_' + str(int(tmid))
            fig.savefig(outfile)
            plt.close()

    #blank figure for make compatibility
    fig, ax = plt.subplots()
        
    outfile = FIG_DIR + 'cloudevents_set_figure.png'
    plt.savefig(outfile)

def get_cloud_intervals(caslwc, cast):
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
    
    t_tuples = []
    casindtuples = []
    #pick peaks with sufficient half-max
    for big_cluster in big_cas_clusters:
        if np.max(caslwc[big_cluster])/2. > 0.7e-5:
            imin = big_cluster[0]
            imax = big_cluster[-1]
            t_tuples.append((cast[imin], cast[imax]))
    return t_tuples

def get_nconc_by_bin(datablock, change_cas_corr, cutoff_bins):
    """
    Returns (nconc_cas, nconc_cdp)
    """
    if cutoff_bins:
        cas_offset = 3
        cdp_offset = 2
    else:
        cas_offset = 0
        cdp_offset = 0
    nconc_cas = []
    nconc_cdp = []
    for i in range(low_bin_cas+cas_offset, high_bin_cas):
        if change_cas_corr:
            nconc_cas.append(np.mean(datablock[:, i]*datablock[:, 3]))
        else:
            nconc_cas.append(np.mean(datablock[:, i]))
    for i in range(low_bin_cdp+cdp_offset, high_bin_cdp):
        nconc_cdp.append(np.mean(datablock[:, i]))
    return(np.array(nconc_cas), np.array(nconc_cdp))

def get_ss_vs_t(datablock, change_cas_corr, cutoff_bins):
    """
    Returns (ss_cas, ss_cdp).
    """
    T = datablock[:, 1] #temp
    w = datablock[:, 2] #vert vel
    one = np.ones(np.shape(T))
    A = g*(L*Ra/(Cp*R)*one/T - one)*1./Ra*one/T

    (nconc_cas, nconc_cdp) = get_nconc_vs_t(datablock, change_cas_corr, cutoff_bins)
    (meanr_cas, meanr_cdp) = get_meanr_vs_t(datablock, change_cas_corr, cutoff_bins)
    ss_cas = A*w/(4*np.pi*D*nconc_cas*meanr_cas)
    ss_cdp = A*w/(4*np.pi*D*nconc_cdp*meanr_cdp)
    #tot_nconc_cas = np.sum(datablock[:, 3:3+nbins_cas], axis=1)
    #meanr_cas = np.dot(datablock[:, 3:3+nbins_cas], \
    #        np.transpose(centr_cas))/tot_nconc_cas
    #ss_cas = 1./(4.*np.pi*D)*A*w/(tot_nconc_cas*meanr_cas)

    #tot_nconc_cdp = np.sum(datablock[:, 3:3+nbins_cdp], axis=1)
    #meanr_cdp = np.dot(datablock[:, 3:3+nbins_cdp], \
    #        np.transpose(centr_cdp))/tot_nconc_cdp
    #ss_cdp = 1./(4.*np.pi*D)*A*w/(tot_nconc_cdp*meanr_cdp)
    return (np.array(ss_cas), np.array(ss_cdp))

def get_vert_wind_vel_vs_t(datablock):
    """
    Returns w.
    """
    w = datablock[:, 2] #vert vel
    return w

def get_A_vs_t(datablock):
    """
    gets A.
    """
    T = datablock[:, 1] #temp
    one = np.ones(np.shape(T))
    A = g*(L*Ra/(Cp*R)*one/T - one)*1./Ra*one/T
    return A

if __name__ == "__main__":
    main()
