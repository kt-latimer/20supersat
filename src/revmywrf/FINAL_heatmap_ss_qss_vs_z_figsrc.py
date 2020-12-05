"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from netCDF4 import Dataset
import numpy as np
import os
import sys

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = [(1, 1, 1), (0.3, 0.3, 0.3)]
cmap = LinearSegmentedColormap.from_list('grey', colors, N=250)
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink =  colors_arr[4]

lwc_filter_val = 1.e-4
w_cutoff = 2

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
    
    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_vs_z(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, \
                                    incl_vent, versionstr)

def make_and_save_ss_qss_vs_z(case_label, case_dir_name, \
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
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #before filtering, get z bins
    z_bins = get_z_bins(z)
    print(z_bins)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
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

    #plot the supersaturations against each other with regression line
    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(32, 12)

    #all points passing filtering criteria
    hist, bins = np.histogram(z, bins=z_bins, density=True)
    z_color_scalars = get_z_color_scalars(hist, z_bins)
    im1 = ax1.imshow(z_color_scalars, cmap=cmap, origin='lower', \
                norm=LogNorm(), aspect='auto', \
                #vmin=np.min(hist), vmax=np.max(hist), \
                extent=[ss_min, ss_max, z_bins[0], z_bins[-1]])
    avg_ss_qss, avg_z = get_avg_ss_qss_and_z(ss_qss, z, z_bins)
    ax1.plot(avg_ss_qss, avg_z, 'o-', color=magma_pink, \
            markeredgecolor=magma_pink, markerfacecolor=magma_pink, \
            linewidth=4)
    #cbar1 = fig.colorbar(im1, ax=ax1)
    #cbar1.set_label(r'$\frac{dn_{points}}{dz}$ (m$^{-1}$)')

    #top 10th percentile (wrt w) of points passing filtering criteria
    up10perc_hist, bins = np.histogram(up10perc_z, bins=z_bins, density=True)
    up10perc_z_color_scalars = get_z_color_scalars(up10perc_hist, z_bins)
    im2 = ax2.imshow(up10perc_z_color_scalars, cmap=cmap, origin='lower', \
                norm=LogNorm(), aspect='auto', \
                #vmin=np.min(hist), vmax=np.max(hist), \
                extent=[ss_min, ss_max, z_bins[0], z_bins[-1]])
    up10perc_avg_ss_qss, up10perc_avg_z = get_avg_ss_qss_and_z(up10perc_ss_qss, \
                                                        up10perc_z, z_bins)
    ax2.plot(up10perc_avg_ss_qss, up10perc_avg_z, 'o-', color=magma_pink, \
            markeredgecolor=magma_pink, markerfacecolor=magma_pink, \
            linewidth=4)

    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical')
    cbar.set_label(r'$\frac{dn_{points}}{dz}$ (m$^{-1}$)')

    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax2.set_xlabel(r'$SS_{QSS}$ (%)')
    ax1.set_ylabel(r'z (m)')
    outfile = FIG_DIR + versionstr + 'FINAL_heatmap_ss_qss_vs_z_' \
            + case_label + '_figure.png'
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

    return np.array(z_bins[:-1])

def get_avg_ss_qss_and_z(ss_qss, z, z_bins):

    avg_ss_qss = np.zeros(np.shape(z_bins)[0] - 1)
    avg_z = np.zeros(np.shape(z_bins)[0] - 1)

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        bin_filter = np.logical_and.reduce((
                        (z > lower_bin_edge), \
                        (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_ss_qss[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_qss_slice = ss_qss[bin_filter]
            z_slice = z[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_slice)
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_qss, avg_z

def get_z_color_scalars(hist, z_bins):
    
    d_z = (z_bins[1] - z_bins[0])/100.
    upper_bin_edge_ind = 1
    n_bins = np.shape(z_bins)[0]
    z = z_bins[0]
    z_color_scalars = np.zeros(np.shape(np.arange(z_bins[0], z_bins[-1], d_z)))

    for i, val in enumerate(z_color_scalars):
        if z >= z_bins[upper_bin_edge_ind] and upper_bin_edge_ind < n_bins:
            upper_bin_edge_ind += 1
        z_color_scalars[i] = hist[upper_bin_edge_ind-1]
        z += d_z

    return np.reshape(z_color_scalars, (-1, 1))
        
if __name__ == "__main__":
    main()
