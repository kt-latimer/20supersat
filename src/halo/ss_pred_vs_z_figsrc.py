"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_vs_t, get_ss_vs_t, \
                              get_full_spectrum_bin_radii, \
                              get_full_spectrum_dict

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

lwc_filter_val = 1.e-4 #10**(-3.5)
w_cutoff = 1

rmax = 102.e-6

#hard-coded based on shared_z_lim computed in wrf/ss_pred_vs_z
z_min = 973.0825
z_max = 4529.355
z_lim = (z_min, z_max)

change_cas_corr = True
cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True

##
## physical constants
##
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

def main():
    
    make_indiv_and_alldates_ss_profiles()

def make_indiv_and_alldates_ss_profiles():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_pred_dict = {'allpts': None, 'up10perc': None}
    w_dict = {'allpts': None, 'up10perc': None}
    z_dict = {'allpts': None, 'up10perc': None}

    for date in good_dates:
        ss_pred, w, z = get_ss_pred_and_w_and_z_data(date)

        if np.shape(ss_pred)[0] != 0:
            ss_pred_dict['allpts'] = add_to_alldates_array(ss_pred, \
                                            ss_pred_dict['allpts'])
            w_dict['allpts'] = add_to_alldates_array(w, \
                                            w_dict['allpts'])
            z_dict['allpts'] = add_to_alldates_array(z, \
                                            z_dict['allpts'])

    ss_pred_dict, w_dict, z_dict = get_up10perc_data(ss_pred_dict, \
                                                    w_dict, z_dict)

    h_z, z_bins = np.histogram(z_dict['allpts'], bins=30, density=True)
    #print(z_bins)

    make_and_save_bipanel_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins)

def add_to_alldates_array(ss_pred, ss_pred_alldates):

    if ss_pred_alldates is None:
        return ss_pred
    else:
        return np.concatenate((ss_pred_alldates, ss_pred))

def get_up10perc_data(ss_pred_dict, w_dict, z_dict):

    w_cutoff = np.percentile(w_dict['allpts'], 90)
    up10perc_inds = w_dict['allpts'] > w_cutoff

    ss_pred_dict['up10perc'] = ss_pred_dict['allpts'][up10perc_inds]
    w_dict['up10perc'] = w_dict['allpts'][up10perc_inds]
    z_dict['up10perc'] = z_dict['allpts'][up10perc_inds]
    #print(z_dict['allpts'])
    #print(z_dict['up10perc'])

    return ss_pred_dict, w_dict, z_dict

def get_ss_pred_and_w_and_z_data(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(cas_dict, \
                                cip_dict, change_cas_corr)

    ss_pred = get_ss_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)
    lwc = get_lwc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, rmax)

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']
    z = adlr_dict['data']['alt']

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    print(date)
    print(np.sum(filter_inds))

    if np.sum(filter_inds) != 0:
        ss_pred = ss_pred[filter_inds]
        w = w[filter_inds]
        z = z[filter_inds]
    else:
        ss_pred = np.array([])
        w = np.array([])
        z = np.array([])

    return ss_pred, w, z

def make_and_save_bipanel_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins):

    #fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig, ax1 = plt.subplots()
    n_pts = {'allpts': 0, 'up10perc': 0}

    for key in ss_pred_dict.keys():
        color = colors_dict[key]
        ss_pred = ss_pred_dict[key]
        z = z_dict[key]
        n_pts[key] = np.shape(ss_pred)[0]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])
        #print(key)

        avg_ss_pred, avg_z, se = get_avg_ss_pred_and_z(ss_pred, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
        #print(avg_z)
        avg_ss_pred = avg_ss_pred[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]
        se = se[notnan_inds]

        ax1.plot(avg_ss_pred, avg_z, linestyle='-', color=color)
        #ax1.plot(avg_ss_pred, avg_z, linestyle='-', marker='o', color=color)
        #ax1.fill_betweenx(avg_z, avg_ss_pred - se, avg_ss_pred + se, \
        #                    color=color, alpha=0.5)
        #ax2.hist(z, bins=z_bins, density=False, orientation='horizontal', \
        #        facecolor=color, alpha=0.8)

    #formatting
    ax1.set_ylim((z_min, z_max))
    #ax1.set_xlim((0, 1.5))
    ax1.yaxis.grid()
    #ax2.set_ylim((z_min, z_max))
    #ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{pred}$ (%)')
    #ax2.set_xlabel(r'$N_{points}$')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    #ax2.xaxis.set_major_formatter(formatter)

    #custom legend
    n_allpts = n_pts['allpts']
    n_up10perc = n_pts['up10perc']
    allpts_line = Line2D([0], [0], color=colors_dict['allpts'], \
                        linewidth=6, linestyle='-')
    up10perc_line = Line2D([0], [0], color=colors_dict['up10perc'], \
                        linewidth=6, linestyle='-')
    plt.legend([allpts_line, up10perc_line], ['WCU (N=' +
                str(n_allpts) + ')', 'Top 10% WCU (N=' + \
                str(n_up10perc) + ')'])

    fig.suptitle('Supersaturation in HALO observations')

    outfile = FIG_DIR + 'ss_pred_vs_z_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_avg_ss_pred_and_z(ss_pred, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_ss_pred = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)
    se = np.zeros(n_bins) #standard error

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        if i == n_bins-1: #last upper bin edge is inclusive
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z <= upper_bin_edge)))
        else: 
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_ss_pred[i] = np.nan
            se[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_pred_slice = ss_pred[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_pred[i] = np.nanmean(ss_pred_slice)
            se[i] = np.nanstd(ss_pred_slice)/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)
            #print(i, avg_ss_pred[i], ss_pred_slice)

    return avg_ss_pred, avg_z, se

if __name__ == "__main__":
    main()
