"""
Same as ``ss_pred_vs_z_figsrc.py`` but showing shading for instrument error
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_lwc_vs_t, get_ss_pred_vs_t, \
                get_full_spectrum_dict, \
                get_ss_coeff, get_ss_coeff_partial_dervs
from phys_consts import *

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

    pres_dict = {'allpts': None, 'up10perc': None}
    ss_pred_dict = {'allpts': None, 'up10perc': None}
    temp_dict = {'allpts': None, 'up10perc': None}
    w_dict = {'allpts': None, 'up10perc': None}
    z_dict = {'allpts': None, 'up10perc': None}

    pres, ss_pred, temp, w, z = get_data(date)

    ss_pred_dict['allpts'] =ss_pred
    pres_dict['allpts'] = pres
    w_dict['allpts'] = w
    temp_dict['allpts'] = temp
    z_dict['allpts'] = z

    pres_dict, ss_pred_dict, temp_dict, w_dict, z_dict = \
        get_up10perc_data(pres_dict, ss_pred_dict, temp_dict, w_dict, z_dict)

    h_z, z_bins = np.histogram(z_dict['allpts'], bins=30, density=True)

    make_and_save_plot(pres_dict, ss_pred_dict, temp_dict, \
                                    w_dict, z_dict, z_bins)

def get_up10perc_data(pres_dict, ss_pred_dict, temp_dict, w_dict, z_dict):

    w_cutoff = np.percentile(w_dict['allpts'], 90)
    up10perc_inds = w_dict['allpts'] > w_cutoff

    pres_dict['up10perc'] = pres_dict['allpts'][up10perc_inds]
    ss_pred_dict['up10perc'] = ss_pred_dict['allpts'][up10perc_inds]
    temp_dict['up10perc'] = temp_dict['allpts'][up10perc_inds]
    w_dict['up10perc'] = w_dict['allpts'][up10perc_inds]
    z_dict['up10perc'] = z_dict['allpts'][up10perc_inds]

    return pres_dict, ss_pred_dict, temp_dict, w_dict, z_dict

def get_data(date):

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    pres = ADLR_dict['data']['pres']
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']
    z = ADLR_dict['data']['alt']
    ss_pred = get_ss_vs_t(ADLR_dict, full_spectrum_dict, change_CAS_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    pres = pres[filter_inds]
    ss_pred = ss_pred[filter_inds]
    temp = temp[filter_inds]
    w = w[filter_inds]
    z = z[filter_inds]

    return pres, ss_pred, temp, w, z

def make_and_save_plot(pres_dict, ss_pred_dict, temp_dict, \
                                    w_dict, z_dict, z_bins):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    n_pts = {'allpts': 0, 'up10perc': 0}

    for key in ss_pred_dict.keys():
        color = colors_dict[key]
        pres = pres_dict[key]
        ss_pred = ss_pred_dict[key]
        temp = temp_dict[key]
        w = w_dict[key]
        z = z_dict[key]
        n_pts[key] = np.shape(ss_pred)[0]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])

        avg_ss_pred, avg_z, se_instr = get_avg_ss_pred_and_z(pres, ss_pred, \
                                                            temp, w, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
        avg_ss_pred = avg_ss_pred[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]
        se_instr = se_instr[notnan_inds]

        ax1.plot(avg_ss_pred, avg_z, linestyle='-', marker='o', color=color)
        ax1.fill_betweenx(avg_z, avg_ss_pred - se_instr, \
                            avg_ss_pred + se_instr, \
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

    outfile = FIG_DIR + 'ss_pred_instr_sigma_vs_z_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_avg_ss_pred_and_z(pres, ss_pred, temp, w, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_ss_pred = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)
    se_instr = np.zeros(n_bins) #standard error

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
            se_instr[i] = np.nan
            avg_z[i] = np.nan
        else:
            pres_slice = pres[bin_filter]
            ss_pred_slice = ss_pred[bin_filter]
            temp_slice = temp[bin_filter]
            w_slice = w[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_pred[i] = np.nanmean(ss_pred_slice)
            se_instr[i] = get_se_instr(pres_slice, ss_pred_slice, \
                                                temp_slice, w_slice)
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_pred, avg_z, se_instr

def get_se_instr(pres, ss_pred, temp, w):

    C = get_ss_coeff(pres, temp)

    sigma_pres = 50 #Pa
    sigma_temp = 0.5 #K
    sigma_w = 0.5 #m/s

    dC_dT, dC_dP = get_ss_coeff_partial_dervs(pres, temp)
    sigma_C = C*np.sqrt((dC_dT*sigma_temp)**2. + (dC_dP*sigma_pres)**2.)

    sigma_ss_pred_arr = ss_pred*np.sqrt((sigma_w/w)**2. + (sigma_C/C)**2. + 0.2**2.)
    sigma_ss_pred = \
        np.sqrt(np.sum(sigma_ss_pred_arr**2.))/np.shape(sigma_ss_pred_arr)[0]

    return sigma_ss_pred

if __name__ == "__main__":
    main()
