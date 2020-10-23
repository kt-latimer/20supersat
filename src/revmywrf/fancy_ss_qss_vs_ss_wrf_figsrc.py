"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
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

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_distb_data_dir = BASE_DIR + 'data/revmywrf/ss_distb/'

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
    print('# pts total: ' + str(np.sum(ss_qss < 200)))
    print('max: ' + str(np.nanmax(ss_qss)))
    print('# pts ss > 2%: ' + str(np.sum(ss_qss > 2)))
   
    print_point_count_per_quadrant(ss_qss, ss_wrf)

    # copied from:
    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(20, 10))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(ss_qss, ss_wrf, c=colors['ss'])
    ax_scatter.plot(ax_scatter.get_xlim(), np.add(b, m*np.array(ax_scatter.get_xlim())), \
                    c=colors['line'], \
                    linestyle='dashed', \
                    linewidth=3, \
                    label=('m = ' + str(np.round(m, decimals=2)) + \
                            ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax_scatter.set_xlabel(r'$SS_{QSS}$ (%)')
    ax_scatter.set_ylabel(r'$SS_{WRF}$ (%)')
    plt.legend(loc=2)

    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([ss_qss, ss_wrf]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(ss_qss, bins=bins, color=colors['ss'])
    ax_histy.hist(ss_wrf, bins=bins, color=colors['ss'], orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    ax_histx.set_title(case_label + ' Supersat - QSS approximation vs true value' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent) \
                    + ', full_ss=' + str(full_ss))
    outfile = FIG_DIR + versionstr + 'fancy_ss_qss_vs_ss_wrf_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

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
