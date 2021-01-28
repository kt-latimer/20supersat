"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
versionstr = 'v20_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors

lwc_filter_val = 1.e-4
w_cutoff = 1 

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -5
ss_max = 20
z_min = -100
z_max = 6500

cutoff_bins = True
incl_rain = True
incl_vent = True
full_ss = True

def main():
    
    ss_qss_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_bins_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        ss_qss_allpts, ss_qss_up10perc, z_allpts, z_up10perc, z_bins = \
                                        get_ss_qss_and_z_data(case_label)

        ss_qss_dict['allpts'][case_label] = ss_qss_allpts
        ss_qss_dict['up10perc'][case_label] = ss_qss_up10perc
        z_dict['allpts'][case_label] = z_allpts
        z_dict['up10perc'][case_label] = z_up10perc
        z_bins_dict[case_label] = z_bins

    make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict['allpts'], \
            z_dict['allpts'], z_bins_dict, colors_arr[3], 'allpts')
    make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict['up10perc'], \
            z_dict['up10perc'], z_bins_dict, colors_arr[7], 'up10perc')

def get_ss_qss_and_z_data(case_label):

    case_dir_name = case_label_dict[case_label]

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
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #before filtering, get z bins
    z_bins = get_z_bins(z)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    #(temp > 0)))
                    (temp > 273)))

    del lwc, temp #for memory

    w_filt = w[filter_inds]
    up10perc_cutoff = np.percentile(w_filt, 90)
    up10perc_inds = np.logical_and.reduce((
                            (filter_inds), \
                            (w > up10perc_cutoff)))

    up10perc_ss_qss = ss_qss[up10perc_inds]
    ss_qss = ss_qss[filter_inds]

    up10perc_z = z[up10perc_inds]
    z = z[filter_inds]

    return ss_qss, up10perc_ss_qss, z, up10perc_z, z_bins

def make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict, z_dict, z_bins_dict, color, label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(18, 12)
    linestyle_str_dict = {'Polluted': '-', 'Unpolluted': '--'}

    for case_label in case_label_dict.keys():
        
        linestyle_str = linestyle_str_dict[case_label]
        ss_qss = ss_qss_dict[case_label]
        z = z_dict[case_label]
        z_bins = z_bins_dict[case_label]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])
        print(label, case_label)

        avg_ss_qss, avg_z, se = get_avg_ss_qss_and_z(ss_qss, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_qss))
        avg_ss_qss = avg_ss_qss[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]

        #print(np.sum(avg_ss_qss*area*qvstar*dz)/np.sum(area*qvstar*dz))
        #continue

        #ax1.fill_betweenx(avg_z, avg_ss_qss + se, avg_ss_qss - se, \
        #                                color=magma_pink, alpha=0.4)
        ax1.plot(avg_ss_qss, avg_z, linestyle=linestyle_str, \
                color=color, linewidth=6, label=case_label) 
        #make histogram with area fraction
        n_xyt = 450*450*84 #cheat code for number of points per altitude slice 
        (counts, bins) = np.histogram(z, bins=z_bins, density=False)
        ax2.hist(z_bins[:-1], bins=z_bins, weights=counts/n_xyt, \
                orientation='horizontal', facecolor=(0, 0, 0, 0.0), \
                edgecolor=color, histtype='stepfilled', linewidth=6, \
                linestyle=linestyle_str, label=case_label)

    #return
    #formatting
    ax1.set_ylim((z_min, z_max))
    ax1.yaxis.grid()
    ax2.set_ylim((z_min, z_max))
    ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    #ax2.set_xlabel(r'$\frac{dn_{points}}{dz}$ (m$^{-1}$)')
    ax2.set_xlabel('Avg area fraction')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax2.xaxis.set_major_formatter(formatter)

    #custom legend
    poll_line = Line2D([0], [0], color=color, \
                        linewidth=6, linestyle='-')
    unpoll_line = Line2D([0], [0], color=color, \
                        linewidth=6, linestyle='--')
    ax2.legend([poll_line, unpoll_line], ['Polluted', 'Unpolluted'])

    outfile = FIG_DIR + versionstr + 'FINAL_bipanel_ss_qss_vs_z_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_z_bins(z):

    n_bins = np.shape(z)[1]
    n_edges = n_bins + 1
    avg_z = np.array([np.mean(z[:, i, :, :]) for i in range(n_bins)])
    z_bins = [] 

    for i in range(1, n_bins):
        layer_geom_mean = np.sqrt(avg_z[i-1]*avg_z[i])
        if layer_geom_mean < z_max:
            z_bins.append(layer_geom_mean)
        else:
            break

    z_bins.insert(0, avg_z[0]*np.sqrt(avg_z[0]/avg_z[1]))

    return np.array(z_bins)

def get_avg_ss_qss_and_z(ss_qss, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_ss_qss = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)
    se = np.zeros(n_bins) #standard error

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
            avg_ss_qss[i] = np.nan
            se[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_qss_slice = ss_qss[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_slice)
            se[i] = np.nanstd(ss_qss_slice)/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_qss, avg_z, se

if __name__ == "__main__":
    main()
