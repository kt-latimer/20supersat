"""
Plot LH CDF vs rmax (where LWC \equiv \sum_{r <= rmax}n(r)*4/3\pi r^3*rho_{h20,
liq}/rho_{air}). For each value of rmax place a marker (used to create gif)
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR, WRF_bin_radii

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]

lwc_cutoff_val = 1.e-4
w_cutoff = 1

rho_w = 1000. #density of water (kg/m^3)

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_lh_cdf(case_label) 

def make_and_save_lh_cdf(case_label):

    print(case_label)
    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get lwc file variables 
    lwc_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_drop_volume_cdf', 'r')
    lwc_vars = lwc_file.variables

    #get relevant physical qtys
    lh = met_vars['LH'][...]
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    #filtering to determine total condensational latent heating
    filter_inds = np.logical_and.reduce((
                    (lh > 0), \
                    (temp > 273)))

    lh_tot = np.sum(lh[filter_inds])

    filter_inds_with_w = np.logical_and.reduce((
                                (filter_inds), \
                                (w > w_cutoff)))

    lh = lh[filter_inds_with_w]

    lwc_dict = {}

    for i in range(1, n_WRF_bins+1):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...]
        lwc_i = 4./3.*np.pi*rho_w*r3n_i/rho_air
        lwc_dict[var_name] = lwc_i[filter_inds_with_w]

    del filter_inds, filter_inds_with_w, lwc_i, \
                r3n_i, rho_air, temp, w #for memory

    #close file for memory
    met_file.close()

    lh_cdf = get_lh_cdf(lh, lh_tot, lwc_dict)

    for imax in range(1, n_WRF_bins+1):
        make_plot_with_radius_marker(imax, lh_cdf, case_label)

def get_lh_cdf(lh, lh_tot, lwc_dict):

    lh_cdf = []

    for i in range(1, n_WRF_bins+1):
        var_key = 'r3n_sum_' + str(i)
        lwc_i = lwc_dict[var_key]

        filter_inds_i = lwc_i > lwc_cutoff_val

        lh_tot_i = np.sum(lh[filter_inds_i])
        lh_cdf.append(lh_tot_i/lh_tot)

    return np.array(lh_cdf)

def make_plot_with_radius_marker(imax, lh_cdf, case_label):

    fig, ax = plt.subplots()

    ax.plot(WRF_bin_radii[:-1]*1.e6, lh_cdf, color=magma_pink)

    rmax = WRF_bin_radii[imax]*1.e6
    ax.plot([rmax], [lh_cdf[imax]], marker='o', color='black', markersize=5)

    ax.set_xscale('log')
    ax.set_xlabel(r'$r_{max}$ ($\mu$m)')
    ax.set_ylabel('(Positive) LH CDF')

    outfile = FIG_DIR + 'lh_cdf_vs_rmax_' + str(imax) \
                    + '_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
