"""
predicted ss versus z with area fraction histogram
"""
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
CSV_DATA_DIR = BASE_DIR + 'data/wrf/'
from wrf.ss_functions import get_lwc, get_ss_pred, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict = {'allpts': colors_arr[3], 'up10perc': colors_arr[7], \
                                            'fan': (0.7, 0.7, 0.7)}

case_label_dict = {'Polluted': 'C_BG/', 'Unpolluted': 'C_PI/'}
set_label_dict = {'allpts': 'WCU', 'up10perc': 'Top 10% WCU'}

ss_min = -5
ss_max = 20
area_frac_cutoff = 1.e-4

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

    fig, ax_arr = plt.subplots(1, 2, sharey=True)

    shared_z_lims = get_shared_z_lims(z_dict, z_bins_dict)
    print(shared_z_lims)

    for i, case_label in enumerate(case_label_dict.keys()):
        z_bins = z_bins_dict[case_label]
        for j, set_label in enumerate(set_label_dict.keys()):
            ss_pred = ss_pred_dict[set_label][case_label]
            z = z_dict[set_label][case_label]
            avg_ss_pred, avg_z, se = get_avg_ss_pred_and_z(ss_pred, z, z_bins)
            #print(case_label, set_label)
            #print(z_bins)

            notnan_inds = np.logical_not(np.isnan(avg_ss_pred))
            #in_z_lim_inds = get_in_z_lim_inds(avg_z, shared_z_lims)
            #avg_ss_pred = avg_ss_pred[np.logical_and(notnan_inds, \
            #                                        in_z_lim_inds)]
            #avg_z = avg_z[np.logical_and(notnan_inds, in_z_lim_inds)]
            #se = se[np.logical_and(notnan_inds, in_z_lim_inds)]

            ax_arr[i].plot(avg_ss_pred, avg_z, color=colors_dict[set_label], \
                                            label=set_label_dict[set_label]) 
            #ax_arr[i].fill_betweenx(avg_z, avg_ss_pred - se, avg_ss_pred + se, \
            #                        color=colors_dict[set_label], alpha=0.5) 
            #make histogram with area fraction
            #ax2.hist(z_bins[:-1], bins=z_bins, weights=counts/n_xyt, \
            #        orientation='horizontal', facecolor=(0, 0, 0, 0.0), \
            #        edgecolor=color, histtype='stepfilled', \
            #        linestyle=linestyle_str, label=case_label)
        ss_wrf_fan, z_fan = get_fan_ss_profile_data(case_label)
        #in_z_lim_inds = get_in_z_lim_inds(z_fan, shared_z_lims)
        #ss_wrf_fan = ss_wrf_fan[in_z_lim_inds]
        #z_fan = z_fan[in_z_lim_inds]
        ax_arr[i].plot(ss_wrf_fan, z_fan, color=colors_dict['fan'], \
                                    label='Fan et al') 

        #formatting
        ax_arr[i].set_ylim(shared_z_lims)
        ax_arr[i].yaxis.grid()
        ax_arr[i].set_xlabel(r'$SS$ (%)')
        ax_arr[i].set_title(case_label)
        #ax2.set_ylim((z_min, z_max))
        #ax2.yaxis.grid()
        #ax2.set_xlabel('Avg area fraction')
        #formatter = ticker.ScalarFormatter(useMathText=True)
        #formatter.set_scientific(True) 
        #formatter.set_powerlimits((-1,1)) 
        #ax2.xaxis.set_major_formatter(formatter)

    ax_arr[0].set_ylabel(r'z (m)')
    handles, labels = ax_arr[1].get_legend_handles_labels()
    ax_arr[1].legend(handles=handles, labels=labels)#, \
                #bbox_to_anchor=(1.04,1), borderaxespad=0)

    fig.suptitle('Supersaturation in WRF simulations')

    outfile = FIG_DIR + 'ss_pred_vs_z_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_shared_z_lims(z_dict, z_bins_dict):

    #z-limits where all four area fractions (combos of poll/unpoll and
    # allpts/up10perc) exceed the area fraction cutoff defined at top

    n_xyt = 450*450*84 #cheat code for number of points per altitude slice 

    z_lims = []
    for case_label in case_label_dict.keys():
        z_bins = z_bins_dict[case_label]
        for set_label in set_label_dict.keys():
            z = z_dict[set_label][case_label]
            (counts, bins) = np.histogram(z, bins=z_bins, density=False)
            area_frac = counts/n_xyt
            z_lims.append(get_z_lim(area_frac, z_bins))

    z_min = max([z_lim[0] for z_lim in z_lims])
    z_max = min([z_lim[1] for z_lim in z_lims])

    return (z_min, z_max)

def get_z_lim(area_frac, z_bins):

    i = 0
    while area_frac[i] < area_frac_cutoff:
        i += 1
    z_min = z_bins[i]
    i_min = i
    while area_frac[i] > area_frac_cutoff:
        i += 1
    z_max = z_bins[i]
    i_max = i
         
    return (z_min, z_max)

def get_in_z_lim_inds(z_vals, z_lim):

    return np.logical_and(z_vals > z_lim[0], z_vals < z_lim[1])

def get_fan_ss_profile_data(case_label):

    #get ss profiles extracted from fig 4 of fan et al 2018

    fan_ss_profile = []
    with open(CSV_DATA_DIR + 'ss_fan_deep_' + case_label + '.csv', 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            fan_ss_profile.append(row)
    fan_ss_profile = np.array(fan_ss_profile)

    #fan data in km ours in m
    return fan_ss_profile[:, 0], fan_ss_profile[:, 1]*1000

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
            se[i] = np.nanstd(ss_pred_slice)#/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_pred, avg_z, se

if __name__ == "__main__":
    main()
