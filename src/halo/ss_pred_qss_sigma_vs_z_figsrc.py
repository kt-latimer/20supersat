"""
Same as ``ss_pred_vs_z_figsrc.py`` but showing shading for SS_QSS error (i.e.
the range of possible SS_WRF values given a certain value of SS_QSS or
equivalently SS_pred) and also including point count histogram
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_vs_t, get_ss_pred_vs_t, \
                        get_full_spectrum_dict, get_grid_ind, \
                        LSR_INT, LSR_SLOPE
from phys_consts import *
from wrf import DATA_DIR as WRF_DATA_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

lwc_filter_val = 1.e-4
w_cutoff = 1

rmax = 102.e-6

z_min = -100
z_max = 6500

change_CAS_corr = True
cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True

def main():
    
    make_ss_profile()

def make_ss_profile():

    ss_pred_dict = {'allpts': None, 'up10perc': None}
    w_dict = {'allpts': None, 'up10perc': None}
    z_dict = {'allpts': None, 'up10perc': None}

    ss_pred, w, z = get_data()

    ss_pred_dict['allpts'] =ss_pred
    w_dict['allpts'] = w
    z_dict['allpts'] = z

    ss_pred_dict, w_dict, z_dict = get_up10perc_data(ss_pred_dict, \
                                                    w_dict, z_dict)

    h_z, z_bins = np.histogram(z_dict['allpts'], bins=30, density=True)

    make_and_save_plot(ss_pred_dict, z_dict, z_bins)

def get_up10perc_data(ss_pred_dict, w_dict, z_dict):

    w_cutoff = np.percentile(w_dict['allpts'], 90)
    up10perc_inds = w_dict['allpts'] > w_cutoff

    ss_pred_dict['up10perc'] = ss_pred_dict['allpts'][up10perc_inds]
    w_dict['up10perc'] = w_dict['allpts'][up10perc_inds]
    z_dict['up10perc'] = z_dict['allpts'][up10perc_inds]

    return ss_pred_dict, w_dict, z_dict

def get_data():

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']
    z = ADLR_dict['data']['alt']
    ss_pred = get_ss_pred_vs_t(ADLR_dict, full_spectrum_dict, change_CAS_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    ss_pred = ss_pred[filter_inds]
    w = w[filter_inds]
    z = z[filter_inds]

    return ss_pred, w, z

def make_and_save_plot(ss_pred_dict, z_dict, z_bins):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    n_pts = {'allpts': 0, 'up10perc': 0}

    ss_pred_domain = get_ss_pred_domain(ss_pred_dict['allpts'])
    ss_qss_sigma = get_ss_qss_sigma(ss_pred_domain)

    for key in ss_pred_dict.keys():
        color = colors_dict[key]
        ss_pred = ss_pred_dict[key]
        z = z_dict[key]
        n_pts[key] = np.shape(ss_pred)[0]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])

        avg_ss_pred, avg_z, se_qss = get_avg_ss_pred_and_z(ss_pred, \
                                ss_pred_domain, ss_qss_sigma, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
        avg_ss_pred = avg_ss_pred[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]
        se_qss = se_qss[notnan_inds]

        ax1.plot(avg_ss_pred, avg_z, linestyle='-', marker='o', color=color)
        ax1.fill_betweenx(avg_z, avg_ss_pred - se_qss, \
                            avg_ss_pred + se_qss, \
                            color=color, alpha=0.5)
        ax2.hist(z, bins=z_bins, density=False, orientation='horizontal', \
                facecolor=color, alpha=0.8)

    #formatting
    ax1.set_ylim((z_min, z_max))
    ax1.set_xlim((0, 1.5))
    ax1.yaxis.grid()
    ax2.set_ylim((z_min, z_max))
    ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{pred}$ (%)')
    ax2.set_xlabel(r'$N_{points}$')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax2.xaxis.set_major_formatter(formatter)

    #custom legend
    n_allpts = n_pts['allpts']
    n_up10perc = n_pts['up10perc']
    allpts_line = Line2D([0], [0], color=colors_dict['allpts'], \
                        linewidth=6, linestyle='-')
    up10perc_line = Line2D([0], [0], color=colors_dict['up10perc'], \
                        linewidth=6, linestyle='-')
    ax2.legend([allpts_line, up10perc_line], ['All cloudy updrafts (N=' +
                str(n_allpts) + ')', 'Top 10% cloudy updrafts (N=' + \
                str(n_up10perc) + ')'])

    fig.suptitle('Supersaturation and area fraction vertical profiles - HALO')

    outfile = FIG_DIR + 'ss_pred_qss_sigma_vs_z_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_avg_ss_pred_and_z(ss_pred, ss_pred_domain, ss_qss_sigma, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_ss_pred = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)
    se_qss = np.zeros(n_bins) #standard error

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
            se_qss[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_pred_slice = ss_pred[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_pred[i] = np.nanmean(ss_pred_slice)
            se_qss[i] = get_se_qss(ss_pred_slice, ss_pred_domain, ss_qss_sigma)
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_pred, avg_z, se_qss

def get_ss_pred_domain(ss_pred):

    min_ss_pred = np.nanmin(ss_pred)
    max_ss_pred = np.nanmax(ss_pred)

    return np.arange(min_ss_pred, max_ss_pred, 0.1)

def get_ss_qss_sigma(ss_pred_domain):

    wrf_data_dict = np.load(WRF_DATA_DIR + 'filtered_data_dict.npy', \
                                            allow_pickle=True).item() 

    ss_qss_poll = wrf_data_dict['Polluted']['ss_qss'] 
    ss_qss_unpoll = wrf_data_dict['Unpolluted']['ss_qss'] 
    ss_wrf_poll = wrf_data_dict['Polluted']['ss_wrf'] 
    ss_wrf_unpoll = wrf_data_dict['Unpolluted']['ss_wrf'] 

    ss_qss = np.concatenate((ss_qss_poll, ss_qss_unpoll))
    ss_pred = LSR_SLOPE*ss_qss + LSR_INT
    ss_wrf = np.concatenate((ss_wrf_poll, ss_wrf_unpoll))

    ss_qss_sigma = np.zeros(np.shape(ss_pred_domain))

    for i, lo_bin_edge in enumerate(ss_pred_domain[:-1]):
        hi_bin_edge = ss_pred_domain[i+1]
        bin_filter = np.logical_and(ss_pred >= lo_bin_edge, \
                                    ss_pred < hi_bin_edge)
        ss_wrf_bin_vals = ss_wrf[bin_filter]
        ss_qss_sigma[i] = np.nanstd(ss_wrf_bin_vals)

    return ss_qss_sigma
         
def get_se_qss(ss_pred, ss_pred_domain, ss_qss_sigma):

    se_qss_arr = np.zeros(np.shape(ss_pred))

    for i, val in enumerate(ss_pred):
        se_qss_arr[i] = get_one_sigma(val, ss_pred_domain, ss_qss_sigma)

    se_qss = np.sqrt(np.sum(se_qss_arr**2.))/np.shape(se_qss_arr)[0]

    return se_qss

def get_one_sigma(val, ss_pred_domain, ss_qss_sigma):
    
    val_ind = get_grid_ind(ss_pred_domain, val)

    return ss_qss_sigma[val_ind]

if __name__ == "__main__":
    main()
