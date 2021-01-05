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

#for plotting
versionstr = 'v4_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
magma = cm.get_cmap('magma')
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

log_lwc_min = -5.5
log_lwc_max = -3.5
n_lwc_vals = 10
lwc_filter_vals = 10**np.linspace(log_lwc_min, log_lwc_max, n_lwc_vals)

w_min = 0.5
w_max = 5.5
n_w_vals = 10
w_filter_vals = np.linspace(w_min, w_max, n_w_vals)

#modified color structure for histograms
magma_discrete = cm.get_cmap('magma', n_lwc_vals*n_w_vals)
colors = magma_discrete.colors 

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    filename = DATA_DIR + 'v3_systematic_filtering_evaluation_data.npy'
    data_dir = np.load(filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        case_data_dir = data_dir[case_label]
        ss_qss_distbs = case_data_dir['ss_qss_distbs']
        ss_wrf_distbs = case_data_dir['ss_wrf_distbs']
        make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label)

def make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label):

    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.set_size_inches(30, 15)

    inds_subset = [1, 4, 9]

    #qss plot
    for i, j in product(inds_subset, inds_subset):
        c_ind = i*n_w_vals + j
        bins = ss_qss_distbs[i, j]['bins']
        counts = ss_qss_distbs[i, j]['counts']
        counts = np.concatenate(([0], counts))
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax1.step(bins, np.log10(counts), where='post', \
                color=colors[c_ind], linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))

    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax1.set_ylabel(r'log($\frac{dn_{points}}{dSS}$)')
    ax1.legend()

    #wrf plot
    for i, j in product(inds_subset, inds_subset):
        c_ind = i*n_w_vals + j
        bins = ss_wrf_distbs[i, j]['bins']
        counts = ss_wrf_distbs[i, j]['counts']
        counts = np.concatenate(([0], counts))
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax2.step(bins, np.log10(counts), where='post', \
                color=colors[c_ind], linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))

    ax2.set_xlabel(r'$SS_{WRF}$ (%)')

    outfile = FIG_DIR + versionstr + 'ss_distb_charts_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def get_ss_bins(ss_min, ss_max):

    lo_end = np.floor(ss_min/d_ss)*d_ss
    hi_end = (np.ceil(ss_max/d_ss) + 1)*d_ss
    ss_bins = np.around(np.arange(lo_end, hi_end, d_ss), dec_prec)

    return ss_bins

if __name__ == "__main__":
    main()
