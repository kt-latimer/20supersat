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
figversionstr = 'v1_'
dataversionstr = 'v7_'
matplotlib.rcParams.update({'font.size': 25})
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
lwc_inds_subset = [4, 5]
w_inds_subset = [1, 2]
magma_discrete = cm.get_cmap('magma', len(lwc_inds_subset)*len(w_inds_subset))
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
    
    filename = dataversionstr + 'systematic_filtering_evaluation_data.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs = \
            get_data_from_dict(case_label, data_dict)
        make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label)
    
    dist_arr = get_dist_arr(data_dict)
    make_and_save_dist_heatmap(dist_arr, case_label)

def get_data_from_dict(case_label, data_dict):

    case_dict = data_dict[case_label]

    m_arr = case_dict['m_arr']
    npts_arr = case_dict['npts_arr']
    rsq_arr = case_dict['rsq_arr']
    ss_qss_distbs = case_dict['ss_qss_distbs']
    ss_wrf_distbs = case_dict['ss_wrf_distbs']

    return m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs

def make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(30, 15)

    #slope plot
    im1 = ax1.imshow(m_arr.T, cmap=magma, vmin=-0.2, vmax=0.9)
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel('Least squares slope')

    ax1.set_xticks(np.arange(n_lwc_vals))
    ax1.set_yticks(np.arange(n_w_vals))
    ax1.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax1.set_yticklabels(np.around(w_filter_vals, 2))
    ax1.set_xbound([-0.5, 6.5])
    ax1.set_ybound([-0.5, -0.5 + n_w_vals])

    ax1.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax1.set_ylabel('Min w cutoff (m/s)')

    #r squared plot
    im2 = ax2.imshow(rsq_arr.T, cmap=magma, vmin=0, vmax=0.9)
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel(r'R$^2$')

    ax2.set_xticks(np.arange(n_lwc_vals))
    ax2.set_yticks(np.arange(n_w_vals))
    ax2.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax2.set_yticklabels(np.around(w_filter_vals, 2))
    ax2.set_xbound([-0.5, 6.5])
    ax2.set_ybound([-0.5, -0.5 + n_w_vals])

    ax2.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    outfile = FIG_DIR + figversionstr + 'FINAL_from_data_regres_param_heatmaps_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def get_dist_arr(data_dict):

    dist_arr = np.zeros(np.shape(data_dict['Polluted']['m_arr']))

    for key1 in ["Polluted", "Unpolluted"]:
        for key2 in ["m_arr", "rsq_arr"]:
            dist_arr += (1 - data_dict[key1][key2])**2.

    return np.sqrt(dist_arr)

def make_and_save_dist_heatmap(dist_arr, case_label):
    """
    adapted from `Creating annotated heatmaps' matplotlib tutorial
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    im = ax.imshow(dist_arr.T, cmap=rev_magma)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Euclidian distance from ideal point (1, 1, 1, 1)')
    annotate_heatmap(im)

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    outfile = FIG_DIR + figversionstr + 'FINAL_from_data_dist_heatmap_figure.png'
    plt.savefig(outfile)
    plt.close()    

def annotate_heatmap(im):

    data = im.get_array()
    threshold = im.norm(data.max())/2.
    valfmt="{x:.2f}"
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    textcolors=["black", "white"]
    kw = dict(horizontalalignment="center", \
              verticalalignment="center")

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

if __name__ == "__main__":
    main()
