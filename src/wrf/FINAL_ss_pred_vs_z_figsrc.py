"""
predicted ss versus z
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
CSV_DATA_DIR = BASE_DIR + 'data/wrf/'
from wrf.ss_functions import get_lwc, get_ss_pred, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict = {'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
linestyle_str_dict = {'Polluted': '-', 'Unpolluted': '--'}
set_label_dict = {'allpts': 'All filtered data points', \
                    'up10perc': 'Top 10 percentiles by w'}

z_min = -100
z_max = 6500

def main():
    
    ss_pred_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_bins_dict = {'Polluted': None, 'Unpolluted': None}

    filtered_data_dict = np.load(DATA_DIR + 'filtered_data_dict.npy', \
                                    allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        case_filtered_data_dict = filtered_data_dict[case_label]
        ss_pred_allpts, ss_pred_up10perc, z_allpts, z_up10perc, z_bins = \
                                get_ss_pred_and_z_data(case_filtered_data_dict)

        ss_pred_dict['allpts'][case_label] = ss_pred_allpts
        ss_pred_dict['up10perc'][case_label] = ss_pred_up10perc
        z_dict['allpts'][case_label] = z_allpts
        z_dict['up10perc'][case_label] = z_up10perc
        z_bins_dict[case_label] = z_bins

    make_and_save_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins_dict)

def get_ss_pred_and_z_data(case_filtered_data_dict):

    ss_qss = case_filtered_data_dict['ss_qss']
    ss_pred = get_ss_pred(ss_qss)
    w = case_filtered_data_dict['w']
    z = case_filtered_data_dict['z']
    z_bins = case_filtered_data_dict['z_bins']

    up10perc_cutoff = np.percentile(w, 90)
    up10perc_inds = w > up10perc_cutoff

    up10perc_ss_pred = ss_pred[up10perc_inds]
    up10perc_z = z[up10perc_inds]

    return ss_pred, up10perc_ss_pred, z, up10perc_z, z_bins

def make_and_save_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins_dict):

    fig, ax = plt.subplots()

    for set_label in set_label_dict.keys():
        for case_label in case_label_dict.keys():
            ss_pred = ss_pred_dict[set_label][case_label]
            z = z_dict[set_label][case_label]
            z_bins = z_bins_dict[case_label]
            dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                            range(np.shape(z_bins)[0] - 1)])

            avg_ss_pred, avg_z, se = get_avg_ss_pred_and_z(ss_pred, z, z_bins)
            notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
            avg_ss_pred = avg_ss_pred[notnan_inds]
            avg_z = avg_z[notnan_inds]
            dz = dz[notnan_inds]

            color = colors_dict[set_label]
            label = set_label_dict[set_label] + ' - ' + case_label
            linestyle_str = linestyle_str_dict[case_label]
            ax.plot(avg_ss_pred, avg_z, linestyle=linestyle_str, \
                                        color=color, label=label)

    ax = plot_fan_ss(ax)

    #formatting
    ax.set_ylim((z_min, z_max))
    ax.yaxis.grid()
    ax.set_xlabel(r'$SS_{pred}$ (%)')
    ax.set_ylabel(r'z (m)')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, \
                bbox_to_anchor=(1.04,1), borderaxespad=0)

    fig.suptitle('Supersaturation and area fraction vertical profiles - WRF')

    outfile = FIG_DIR + 'FINAL_ss_pred_vs_z_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def plot_fan_ss(ax):

    #get ss profiles extracted from fig 4 of fan et al 2018
    #kinda shoddy code

    #profile_filenames = ['deep_poll', 'deep_unpoll', 'warm_poll', 'warm_unpoll']
    profile_filenames = ['deep_poll', 'deep_unpoll']
    profile_linestyles = ['-', '--', '-', '--']
    profile_colors = [(0.3, 0.3, 0.3), (0.3, 0.3, 0.3), \
                        (0.7, 0.7, 0.7), (0.7, 0.7, 0.7)] 
    profile_labels = ['Fan et al: "Deep cloud" Polluted', 'Fan et al: "Deep ' \
                      + 'cloud" Unpolluted', 'Fan et al: "Warm cloud" Polluted',\
                      'Fan et al: "Warm cloud" Unpolluted']

    for i, profile_filename in enumerate(profile_filenames):
        fan_ss_profile = []
        with open(CSV_DATA_DIR + 'ss_fan_' + profile_filename + '.csv', 'r') as readFile:
            csvreader = csv.reader(readFile, \
                    quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
            for row in csvreader:
                fan_ss_profile.append(row)
        #fan data in km ours in m
        fan_ss_profile = np.array(fan_ss_profile) 
        ax.plot(fan_ss_profile[:, 0], \
                fan_ss_profile[:, 1]*1000, \
                linestyle=profile_linestyles[i], \
                color=profile_colors[i], \
                label=profile_labels[i])

    return ax

def get_avg_ss_pred_and_z(ss_pred, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_ss_pred = np.zeros(n_bins)
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
            avg_ss_pred[i] = np.nan
            se[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_pred_slice = ss_pred[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_pred[i] = np.nanmean(ss_pred_slice)
            se[i] = np.nanstd(ss_pred_slice)/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_pred, avg_z, se

if __name__ == "__main__":
    main()
