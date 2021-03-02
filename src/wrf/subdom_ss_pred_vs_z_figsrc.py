"""
predicted ss versus z with area fraction histogram
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

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -5
ss_max = 20
z_min = -100
z_max = 6500

lon_min = -60.79
lat_min = -3.28
lon_max = -60.47
lat_max = -2.86
R_e = 6.3781e6 #radius of Earth (m)

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

    make_and_save_ss_pred_vs_z(ss_pred_dict['allpts'], \
            z_dict['allpts'], z_bins_dict, colors_arr[3], 'allpts')
    make_and_save_ss_pred_vs_z(ss_pred_dict['up10perc'], \
            z_dict['up10perc'], z_bins_dict, colors_arr[7], 'up10perc')

def get_ss_pred_and_z_data(case_filtered_data_dict):

    ss_qss = case_filtered_data_dict['ss_qss']
    ss_pred = get_ss_pred(ss_qss)
    w = case_filtered_data_dict['w']
    x = case_filtered_data_dict['x']
    y = case_filtered_data_dict['y']
    lon = x*180./np.pi*1./R_e
    lat = y*180./np.pi*1./R_e
    z = case_filtered_data_dict['z']
    z_bins = case_filtered_data_dict['z_bins']
    
    subdom_inds = np.logical_and.reduce((
                        (lon > lon_min), \
                        (lat > lat_min), \
                        (lon < lon_max), \
                        (lat < lat_max)))

    ss_pred = ss_pred[subdom_inds]
    w = w[subdom_inds]
    z = z[subdom_inds]

    up10perc_cutoff = np.percentile(w, 90)
    up10perc_inds = (w > up10perc_cutoff)

    up10perc_ss_pred = ss_pred[up10perc_inds]
    up10perc_z = z[up10perc_inds]

    return ss_pred, up10perc_ss_pred, z, up10perc_z, z_bins

def make_and_save_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins_dict, color, label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    linestyle_str_dict = {'Polluted': '-', 'Unpolluted': '--'}

    if label == 'up10perc':
        ax1 = plot_fan_ss(ax1)

    for case_label in case_label_dict.keys():
        
        linestyle_str = linestyle_str_dict[case_label]
        ss_pred = ss_pred_dict[case_label]
        z = z_dict[case_label]
        z_bins = z_bins_dict[case_label]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])
        print(label, case_label)

        avg_ss_pred, avg_z, se = get_avg_ss_pred_and_z(ss_pred, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
        avg_ss_pred = avg_ss_pred[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]

        ax1.plot(avg_ss_pred, avg_z, linestyle=linestyle_str, \
                color=color, label=case_label) 
        #make histogram with area fraction
        n_xyt = 450*450*84 #cheat code for number of points per altitude slice 
        (counts, bins) = np.histogram(z, bins=z_bins, density=False)
        ax2.hist(z_bins[:-1], bins=z_bins, weights=counts/n_xyt, \
                orientation='horizontal', facecolor=(0, 0, 0, 0.0), \
                edgecolor=color, histtype='stepfilled', \
                linestyle=linestyle_str, label=case_label)

    #formatting
    ax1.set_ylim((z_min, z_max))
    ax1.yaxis.grid()
    ax2.set_ylim((z_min, z_max))
    ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{pred}$ (%)')
    ax2.set_xlabel('Avg area fraction')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax2.xaxis.set_major_formatter(formatter)

    handles, labels = ax1.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, \
                bbox_to_anchor=(1.04,1), borderaxespad=0)

    fig.suptitle('Supersaturation and area fraction vertical profiles - WRF ' + case_label)

    outfile = FIG_DIR + 'ss_pred_vs_z_' + label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def plot_fan_ss(ax):

    #get ss profiles extracted from fig 4 of fan et al 2018
    #kinda shoddy code

    profile_filenames = ['deep_poll', 'deep_unpoll', 'warm_poll', 'warm_unpoll']
    profile_linestyles = ['-', '--', '-', '--']
    profile_colors = [(0.3, 0.3, 0.3), (0.3, 0.3, 0.3), \
                        (0.7, 0.7, 0.7), (0.7, 0.7, 0.7)] 
    profile_labels = ['Fan et al: "Deep cloud" Polluted', 'Fan et al: "Deep' \
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
