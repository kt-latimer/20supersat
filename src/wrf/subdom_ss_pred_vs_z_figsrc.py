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

from wrf import DATA_DIR, FIG_DIR
from wrf.ss_pred_functions import get_lwc, get_ss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors

lwc_filter_val = 1.e-4
w_cutoff = 1 
lon_min = -60.79
lat_min = -3.28
lon_max = -60.47
lat_max = -2.86
R_e = 6.3781e6 #radius of Earth (m)

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
    
    ss_pred_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_bins_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        ss_pred_allpts, ss_pred_up10perc, z_allpts, z_up10perc, z_bins = \
                                        get_ss_pred_and_z_data(case_label)

        ss_pred_dict['allpts'][case_label] = ss_pred_allpts
        ss_pred_dict['up10perc'][case_label] = ss_pred_up10perc
        z_dict['allpts'][case_label] = z_allpts
        z_dict['up10perc'][case_label] = z_up10perc
        z_bins_dict[case_label] = z_bins

    make_and_save_bipanel_ss_pred_vs_z(ss_pred_dict['allpts'], \
            z_dict['allpts'], z_bins_dict, colors_arr[3], 'allpts')
    make_and_save_bipanel_ss_pred_vs_z(ss_pred_dict['up10perc'], \
            z_dict['up10perc'], z_bins_dict, colors_arr[7], 'up10perc')

def get_ss_pred_and_z_data(case_label):

    case_dir_name = case_label_dict[case_label]

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
    x = met_vars['x'][...]
    lon = x*180./np.pi*1./R_e
    lon = np.transpose(np.tile(lon, [66, 1, 1, 1]), [1, 0, 2, 3])
    y = met_vars['y'][...]
    lat = y*180./np.pi*1./R_e
    lat = np.transpose(np.tile(lat, [66, 1, 1, 1]), [1, 0, 2, 3])
    z = met_vars['z'][...]
    ss_pred = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files / delete vars for memory
    del x, y
    met_file.close()
    dsdsum_file.close()

    #before filtering, get z bins
    z_bins = get_z_bins(z)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (lon > lon_min), \
                    (lat > lat_min), \
                    (lon < lon_max), \
                    (lat < lat_max), \
                    (temp > 273)))

    del lat, lon, lwc, temp #for memory

    w_filt = w[filter_inds]
    up10perc_cutoff = np.percentile(w_filt, 90)
    up10perc_inds = np.logical_and.reduce((
                            (filter_inds), \
                            (w > up10perc_cutoff)))

    up10perc_ss_pred = ss_pred[up10perc_inds]
    ss_pred = ss_pred[filter_inds]

    up10perc_z = z[up10perc_inds]
    z = z[filter_inds]

    return ss_pred, up10perc_ss_pred, z, up10perc_z, z_bins

def make_and_save_bipanel_ss_pred_vs_z(ss_pred_dict, z_dict, z_bins_dict, color, label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    linestyle_str_dict = {'Polluted': '-', 'Unpolluted': '--'}

    for case_label in case_label_dict.keys():
        
        linestyle_str = linestyle_str_dict[case_label]
        ss_pred = ss_pred_dict[case_label]
        z = z_dict[case_label]
        z_bins = z_bins_dict[case_label]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])

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

    fig.suptitle('Supersaturation and area fraction vertical profiles - WRF ' + case_label \
                    + '\n (Restricted to horizontal subdomain from reference [1])')

    outfile = FIG_DIR + 'subdom_ss_pred_vs_z_' + label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
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
