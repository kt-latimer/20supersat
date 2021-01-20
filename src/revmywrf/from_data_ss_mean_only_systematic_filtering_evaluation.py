"""
1) make heatmap showing least squares slope and r-squared of ss_wrf vs ss_qss
for different vals of w and LWC cutoffs; 2) save ss_wrf and ss_qss
distributions (histograms) in the same parameter space
"""
from itertools import product
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/revmywrf/'
FIG_DIR = BASE_DIR + 'figures/revmywrf/'
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
rev_magma = cm.get_cmap('magma_r')
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

log_lwc_min = -6
log_lwc_max = -2
n_lwc_vals = 9
lwc_filter_vals = 10**np.linspace(log_lwc_min, log_lwc_max, n_lwc_vals)

w_min = 0
w_max = 6 
n_w_vals = 7
w_filter_vals = np.linspace(w_min, w_max, n_w_vals)

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    filename = versionstr + 'ss_mean_only_systematic_filtering_evaluation_data.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        ss_qss_mean_arr, ss_wrf_mean_arr = \
            get_data_from_dict(case_label, data_dict)
        make_and_save_ss_mean_heatmap(ss_qss_mean_arr, \
                            ss_wrf_mean_arr, case_label)

def get_data_from_dict(case_label, data_dict):

    case_dict = data_dict[case_label]

    ss_qss_mean_arr = case_dict['ss_qss_mean_arr']
    ss_wrf_mean_arr = case_dict['ss_wrf_mean_arr']

    return ss_qss_mean_arr, ss_wrf_mean_arr

def make_and_save_ss_mean_heatmap(ss_qss_mean_arr, ss_wrf_mean_arr, case_label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(30, 15)

    im1 = ax1.imshow(ss_qss_mean_arr.T, cmap=rev_magma)
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel(r'$SS_{QSS}$ (%)')

    ax1.set_xticks(np.arange(n_lwc_vals))
    ax1.set_yticks(np.arange(n_w_vals))
    ax1.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax1.set_yticklabels(np.around(w_filter_vals, 2))
    ax1.set_xbound([-0.5, 6.5])
    ax1.set_ybound([-0.5, -0.5 + n_w_vals])

    ax1.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax1.set_ylabel('Min w cutoff (m/s)')

    im2 = ax2.imshow(ss_qss_mean_arr.T, cmap=rev_magma)
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel(r'$SS_{WRF}$ (%)')

    ax2.set_xticks(np.arange(n_lwc_vals))
    ax2.set_yticks(np.arange(n_w_vals))
    ax2.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax2.set_yticklabels(np.around(w_filter_vals, 2))
    ax2.set_xbound([-0.5, 6.5])
    ax2.set_ybound([-0.5, -0.5 + n_w_vals])

    ax2.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    outfile = FIG_DIR + versionstr + 'from_data_ss_mean_heatmap_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
