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
versionstr = 'v7_'
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
    
    filename = versionstr + 'systematic_filtering_evaluation_data.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs = \
            get_data_from_dict(case_label, data_dict)
        make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label)
        make_and_save_npts_heatmap(npts_arr, case_label)
        make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label)

def get_data_from_dict(case_label, data_dict):

    case_dict = data_dict[case_label]

    m_arr = case_dict['m_arr']
    npts_arr = case_dict['npts_arr']
    rsq_arr = case_dict['rsq_arr']
    ss_qss_distbs = case_dict['ss_qss_distbs']
    ss_wrf_distbs = case_dict['ss_wrf_distbs']

    return m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs

def make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label):

    #m_arr = m_arr[:-2]
    #rsq_arr = rsq_arr[:-2]

    #for i, j in product(range(n_lwc_vals), range(n_w_vals)):
        #lwc_filter_val = lwc_filter_vals[i]
        #w_filter_val = w_filter_vals[j]
        #print(i, j)
        #print(lwc_filter_val, w_filter_val)
        #print(m_arr[i, j])

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(45, 15)

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

    #combined plot
    dist_arr = np.sqrt((1.-m_arr)**2. + (1.-rsq_arr)**2.)
    #print(case_label)
    #for i,j in product(range(n_lwc_vals), range(n_w_vals)):
    #    print(lwc_filter_vals[i], w_filter_vals[j], dist_arr[i, j])
    im3 = ax3.imshow(dist_arr.T, cmap=rev_magma, vmin=0.1, vmax=1.5)
    cbar3 = ax3.figure.colorbar(im3, ax=ax3)
    cbar3.ax.set_ylabel('$\sqrt{(1-m)^2 + (1-R^2)^2}$')

    ax3.set_xticks(np.arange(n_lwc_vals))
    ax3.set_yticks(np.arange(n_w_vals))
    ax3.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax3.set_yticklabels(np.around(w_filter_vals, 2))
    ax3.set_xbound([-0.5, 6.5])
    ax3.set_ybound([-0.5, -0.5 + n_w_vals])

    ax3.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    outfile = FIG_DIR + versionstr + 'from_data_regres_param_heatmaps_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def make_and_save_npts_heatmap(npts_arr, case_label):

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    im = ax.imshow(npts_arr.T, cmap=rev_magma, vmin=0, vmax=0.025)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Fraction of points remaining')

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    outfile = FIG_DIR + versionstr + 'from_data_npts_heatmap_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label):

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
    ax1.legend()

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

    ax2.set_xlabel(r'$SS_{WRF}$ (%)')

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
