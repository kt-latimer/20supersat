"""
1) make heatmap showing least squares slope and r-squared of ss_wrf vs ss_qss
for different vals of w and LWC cutoffs; 2) save ss_wrf and ss_qss
distributions (histograms) in the same parameter space
"""
from itertools import product
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
versionstr = 'v3_'
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

data_dict = {'Polluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}, 
             'Unpolluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}}

def main():
    
    for case_label in case_label_dict.keys():
        m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs = \
            get_filter_dependent_values(case_label, \
                case_label_dict[case_label], \
                cutoff_bins, full_ss, incl_rain, \
                incl_vent, versionstr)
        make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label)
        make_and_save_npts_heatmap(npts_arr, case_label)
        make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label)
        
        data_dict[case_label]['m_arr'] = m_arr
        data_dict[case_label]['npts_arr'] = npts_arr
        data_dict[case_label]['rsq_arr'] = rsq_arr
        data_dict[case_label]['ss_qss_distbs'] = ss_qss_distbs 
        data_dict[case_label]['ss_wrf_distbs'] = ss_wrf_distbs
        
    filename = versionstr + 'systematic_filtering_evaluation_data.npy'
    np.save(DATA_DIR + filename, data_dict)

def get_filter_dependent_values(case_label, case_dir_name, \
                                cutoff_bins, full_ss, \
                                incl_rain, incl_vent, versionstr):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100
    shape = np.shape(lwc)
    ntot = shape[0]*shape[1]*shape[2]*shape[3]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    m_arr = np.zeros((n_lwc_vals, n_w_vals))
    npts_arr = np.zeros((n_lwc_vals, n_w_vals))
    rsq_arr = np.zeros((n_lwc_vals, n_w_vals))
    ss_wrf_distbs = np.empty((n_lwc_vals, n_w_vals), dtype=dict) 
    ss_qss_distbs = np.empty((n_lwc_vals, n_w_vals), dtype=dict) 

    for i, j in product(range(n_lwc_vals), range(n_w_vals)):
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        
        #apply filtering criteria
        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_filter_val), \
                        (temp > 273)))

        ss_qss_filt = ss_qss[filter_inds]
        ss_wrf_filt = ss_wrf[filter_inds]

        m, b, R, sig = linregress(ss_qss_filt, ss_wrf_filt)
        npts = np.shape(ss_qss_filt)[0]

        m_arr[i, j] = m
        npts_arr[i, j] = npts/ntot
        rsq_arr[i, j] = R**2.

        ss_qss_bins = get_ss_bins(np.nanmin(ss_qss_filt), \
                                    np.nanmax(ss_qss_filt))
        ss_qss_counts, ss_qss_bins = np.histogram(ss_qss_filt, \
                            bins=ss_qss_bins, density=True)
        ss_wrf_bins = get_ss_bins(np.nanmin(ss_wrf_filt), \
                                    np.nanmax(ss_wrf_filt))
        ss_wrf_counts, ss_wrf_bins = np.histogram(ss_wrf_filt, \
                            bins=ss_wrf_bins, density=True)

        ss_qss_distbs[i, j] = {'counts': ss_qss_counts, 'bins': ss_qss_bins}
        ss_wrf_distbs[i, j] = {'counts': ss_wrf_counts, 'bins': ss_wrf_bins}

    return m_arr, npts_arr, rsq_arr, ss_qss_distbs, ss_wrf_distbs

def make_and_save_regres_param_heatmaps(m_arr, rsq_arr, case_label):

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(45, 15)

    #slope plot
    im1 = ax1.imshow(m_arr, cmap=magma)
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel('Least squares slope')

    ax1.set_xticks(np.arange(n_lwc_vals))
    ax1.set_yticks(np.arange(n_w_vals))
    ax1.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax1.set_yticklabels(np.around(w_filter_vals, 2))

    ax1.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax1.set_ylabel('Min w cutoff (m/s)')

    #r squared plot
    im2 = ax2.imshow(rsq_arr, cmap=magma)
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel(r'R$^2$')

    ax2.set_xticks(np.arange(n_lwc_vals))
    ax2.set_yticks(np.arange(n_w_vals))
    ax2.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax2.set_yticklabels(np.around(w_filter_vals, 2))

    ax2.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    #combined plot
    dist_arr = np.sqrt((1.-m_arr)**2. + (1.-rsq_arr)**2.)
    im3 = ax3.imshow(dist_arr, cmap=magma)
    cbar3 = ax3.figure.colorbar(im3, ax=ax3)
    cbar3.ax.set_ylabel('$\sqrt{(1-m)^2 + (1-R^2)^2}$')

    ax3.set_xticks(np.arange(n_lwc_vals))
    ax3.set_yticks(np.arange(n_w_vals))
    ax3.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax3.set_yticklabels(np.around(w_filter_vals, 2))

    ax3.set_xlabel('Min log(LWC) cutoff (kg/kg)')

    outfile = FIG_DIR + versionstr + 'regres_param_heatmaps_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def make_and_save_npts_heatmap(npts_arr, case_label):

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    im = ax.imshow(npts_arr, cmap=magma)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Fraction of points remaining')

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    outfile = FIG_DIR + versionstr + 'npts_heatmap_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def make_and_save_ss_distb_charts(ss_qss_distbs, ss_wrf_distbs, case_label):

    fig, [ax1, ax2] = plt.subplots(1, 2)
    fig.set_size_inches(30, 15)

    inds_subset = [1, 4, 9]

    #qss plot
    for i, j in product(inds_subset, inds_subset):
        c_ind = i*n_w_vals + j
        bins = ss_qss_distbs[i, j]['bins']
        counts = ss_qss_distbs[i, j]['counts']
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax1.hist(bins[:-1], bins=bins, weights=np.log10(counts), \
                facecolor=(0, 0, 0, 0.0), edgecolor=colors[c_ind], \
                histtype='stepfilled', linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))

    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax1.set_ylabel(r'log($\frac{dn_{points}}{dSS}$)')

    #wrf plot
    for i, j in product(inds_subset, inds_subset):
        c_ind = i*n_w_vals + j
        bins = ss_wrf_distbs[i, j]['bins']
        counts = ss_wrf_distbs[i, j]['counts']
        lwc_filter_val = lwc_filter_vals[i]
        w_filter_val = w_filter_vals[j]
        ax2.hist(bins[:-1], bins=bins, weights=np.log10(counts), \
                facecolor=(0, 0, 0, 0.0), edgecolor=colors[c_ind], \
                histtype='stepfilled', linewidth=4, label='log(LWC) > ' + \
                str(np.round(np.log10(lwc_filter_val), 2)) + ', w > ' + \
                str(np.round(w_filter_val, 2)))

    ax2.set_xlabel(r'$SS_{WRF}$ (%)')
    ax2.legend()

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
