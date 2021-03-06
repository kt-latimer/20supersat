"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
#versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
yellow_to_purple = LinearSegmentedColormap.from_list('white_to_purple', [ \
    (0, '#fde624'), \
    (1, '#440053'), \
    ], N=256)
#white_to_purple = LinearSegmentedColormap.from_list('white_to_purple', [ \
#   (0, '#ffffff'), \
#   (1, '#440053'), \
#], N=256)
                            
lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_distb_data_dir = BASE_DIR + 'data/revmywrf/ss_distb/'

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

#change_dsd_corr = False
#cutoff_bins = False 
#incl_rain = False
#incl_vent = False
#full_ss = False

def main():
    
    versionnum = int(os.environ['versionnum'])
    (dummy_bool, cutoff_bins, full_ss, \
        incl_rain, incl_vent) = get_boolean_params(versionnum)
    versionstr = 'v' + str(versionnum) + '_'

    for case_label in case_label_dict.keys():
        #if case_label == 'Polluted':
        #    continue
        make_and_save_ss_qss_vs_ss_wrf(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_qss_vs_ss_wrf(case_label, case_dir_name, \
                                cutoff_bins, full_ss, \
                                incl_rain, incl_vent, versionstr):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    #(np.abs(w) > w_cutoff), \
                    (w > w_cutoff), \
                    (temp > 273)))
                    #(temp > -2730)))

    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    m, b, R, sig = linregress(ss_qss, ss_wrf)
    print(m, b, R**2)
    print(case_label)
    N_points = np.sum(ss_qss < 200)
    print('# pts total: ' + str(N_points))
    print('max: ' + str(np.nanmax(ss_qss)))
    print('# pts ss > 2%: ' + str(np.sum(ss_qss > 2)))
   
    print_point_count_per_quadrant(ss_qss, ss_wrf)

    #def using_mpl_scatter_density(fig, x, y):
    #    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #    density = ax.scatter_density(x, y, \
    #        norm=matplotlib.colors.LogNorm(vmin=1.e-100, vmax=6.e5), \
    #        cmap=white_to_purple)
    #    fig.colorbar(density, label='Number of points per pixel')
    #    return ax

    #fig = plt.figure(figsize=(21, 12))
    #ax = using_mpl_scatter_density(fig, ss_qss, ss_wrf)

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    
    ss_bins = get_ss_bins(ss_min, ss_max, d_ss)

    h = ax.hist2d(ss_qss, ss_wrf, bins=ss_bins, cmin=1./(N_points*d_ss**2.), \
        density=True, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.plasma)
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label(r'$\frac{d^2n_{points}}{dSS_{QSS}dSS_{WRF}}$')

    ax.set_xlim((ss_min, ss_max))
    ax.set_ylim((ss_min, ss_max))
    ax.set_aspect('equal', 'box')

    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c=colors['line'], \
            linestyle='dashed', \
            linewidth=3, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))

    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'$SS_{WRF}$ (%)')
    plt.legend(loc=2)

    outfile = FIG_DIR + versionstr + 'heatmap_ss_qss_vs_ss_wrf_' \
            + case_label + '_figure_6.png'
    plt.savefig(outfile)
    plt.close()    

def get_ss_bins(ss_min, ss_max, d_ss):

    ss_bins = np.arange(ss_min, ss_max, d_ss)

    return ss_bins

def get_boolean_params(versionnum):

    versionnum = versionnum - 1 #for modular arithmetic
    
    if versionnum > 11:
       versionnum = versionnum - 12

    #here this is a dummy var but keeping for consistency with
    #revhalo for now aka for my sanity
    if versionnum < 12:
        change_cas_corr = False
    else:
        change_cas_corr = True

    if versionnum % 12 < 6:
        cutoff_bins = False
    else:
        cutoff_bins = True

    if versionnum % 6 < 3:
        full_ss = False
    else:
        full_ss = True

    if versionnum % 3 == 0:
        incl_rain = False
    else:
        incl_rain = True

    if versionnum % 3 == 2:
        incl_vent = True
    else:
        incl_vent = False

    return (change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

def print_point_count_per_quadrant(ss_qss, ss_wrf):

    n_q1 = np.sum(np.logical_and(ss_qss > 0, ss_wrf > 0))
    n_q2 = np.sum(np.logical_and(ss_qss < 0, ss_wrf > 0))
    n_q3 = np.sum(np.logical_and(ss_qss < 0, ss_wrf < 0))
    n_q4 = np.sum(np.logical_and(ss_qss > 0, ss_wrf < 0))
    
    print('Number of points in Q1:', n_q1)
    print('Number of points in Q2:', n_q2)
    print('Number of points in Q3:', n_q3)
    print('Number of points in Q4:', n_q4)
    print()
    print('Domain:', np.min(ss_qss), np.max(ss_qss))
    print('Range:', np.min(ss_wrf), np.max(ss_wrf))

if __name__ == "__main__":
    main()
