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
versionstr = 'v8_'
filterset_versionstr = 'v7_'
fullset_versionstr = 'v8_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
magma = cm.get_cmap('magma')
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

#modified color structure for histograms
lwc_inds_subset = [0, 4, 5]
w_inds_subset = [1, 2]
magma_discrete = cm.get_cmap('magma', len(lwc_inds_subset)*len(w_inds_subset)+1)
colors = magma_discrete.colors 

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

data_dict = {'Polluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}, 
             'Unpolluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}}

def main():
    
    filterset_filename = filterset_versionstr + \
        'systematic_filtering_evaluation_data.npy'
    fullset_filename = fullset_versionstr + \
        'systematic_filtering_evaluation_data.npy'
    filterset_data_dict = np.load(DATA_DIR + \
        filterset_filename, allow_pickle=True).item()
    fullset_data_dict = np.load(DATA_DIR + \
        fullset_filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        make_and_save_ss_distb_charts(filterset_data_dict, \
                                fullset_data_dict, case_label)

def make_and_save_ss_distb_charts(filterset_data_dict, \
                            fullset_data_dict, case_label):

    ss_qss_distbs = filterset_data_dict[case_label]['ss_qss_distbs']
    ss_wrf_distbs = filterset_data_dict[case_label]['ss_wrf_distbs']

    full_ss_qss_distbs = fullset_data_dict[case_label]['ss_qss_distbs']
    full_ss_wrf_distbs = fullset_data_dict[case_label]['ss_wrf_distbs']

    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.set_size_inches(30, 15)

    #qss plot
    c_ind = 0
    for i, j in product(lwc_inds_subset, w_inds_subset):
        bins = ss_qss_distbs[i, j]['bins']
        counts = ss_qss_distbs[i, j]['counts']
        counts = np.concatenate(([0], counts))
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax1.step(bins, np.log10(counts), where='post', \
                color=colors[c_ind], linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))
        c_ind += 1
    
    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax1.set_ylabel(r'log($\frac{dn_{points}}{dSS}$)')

    #wrf plot
    c_ind = 0
    for i, j in product(lwc_inds_subset, w_inds_subset):
        bins = ss_wrf_distbs[i, j]['bins']
        counts = ss_wrf_distbs[i, j]['counts']
        counts = np.concatenate(([0], counts))
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax2.step(bins, np.log10(counts), where='post', \
                color=colors[c_ind], linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))
        c_ind += 1
    
    bins = full_ss_wrf_distbs[0, 0]['bins']
    counts = full_ss_wrf_distbs[0, 0]['counts']
    counts = np.concatenate(([0], counts))
    ax2.step(bins, np.log10(counts), where='post', \
            color='g', linewidth=4, \
            label='No LWC cutoff, no w cutoff')

    ax2.set_xlabel(r'$SS_{WRF}$ (%)')

    handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles, labels)

    outfile = FIG_DIR + versionstr + 'from_data_ss_distb_charts_' \
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
