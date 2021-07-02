"""
make smaller data files for quantities after applying LWC, vertical wind
velocity, and temperature filters
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[1]
magma_pink = colors_arr[5]
magma_orange = colors_arr[9]

###
### bin sizes and regime params
###
bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

rho_w = 1000. #density of water (kg/m^3)

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

lwc_cutoff_vals = [-1, 1.e-5, 1.e-4]
w_cutoff_val = 1

def main():
    
    for case_label in case_label_dict.keys():
        lwc_cdf_dict = get_lwc_cdf_dict(case_label)
        make_lwc_cdf(case_label, lwc_cdf_dict)

def get_lwc_cdf_dict(case_label):

    print(case_label)
    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get lwc file variables 
    lwc_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_lwc_spectrum_vars', 'r')
    lwc_vars = lwc_file.variables

    #get relevant physical qtys
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    #filtering to determine total condensational latent heating
    filter_inds = np.logical_and.reduce((
                    (w > w_cutoff_val), \
                    (temp > 273)))

    del temp, w #for memory

    lwc_tot = get_lwc_tot(lwc_vars, rho_air)
    lwc = np.zeros(np.shape(lwc_tot))
    lwc_cdf_dict = {}

    for i in range(1, 34):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...]
        lwc += 4./3.*np.pi*rho_w*r3n_i/rho_air
        for lwc_cutoff_val in lwc_cutoff_vals:
            lwc_cdf_dict = update_lwc_cdf_dict(filter_inds, lwc, \
                            lwc_cdf_dict, lwc_cutoff_val, lwc_tot)

    del filter_inds, lwc_tot, lwc #for memory

    #close file for memory
    met_file.close()

    return lwc_cdf_dict
             
def update_lwc_cdf_dict(filter_inds, lwc, lwc_cdf_dict, \
                                lwc_cutoff_val, lwc_tot):

    filter_inds = np.logical_and(lwc > lwc_cutoff_val, filter_inds)
    lwc_cutoff_key = str(lwc_cutoff_val)

    if lwc_cutoff_key not in lwc_cdf_dict.keys():
        lwc_cdf_dict[lwc_cutoff_key] = {'mean': [], 'std': [], \
                    'median': [], 'lo_quart': [], 'up_quart': []}

    if np.sum(filter_inds) != 0:
        lwc_cdf_dict[lwc_cutoff_key]['mean'].append( \
            np.nanmean(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['std'].append( \
            np.nanstd(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['median'].append( \
            np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['lo_quart'].append( \
            (np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds]) - \
            np.percentile(lwc[filter_inds]/lwc_tot[filter_inds], 25)))
        lwc_cdf_dict[lwc_cutoff_key]['up_quart'].append( \
            (np.percentile(lwc[filter_inds]/lwc_tot[filter_inds], 75) - \
            np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds])))
    else:
        lwc_cdf_dict[lwc_cutoff_key]['mean'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['std'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['median'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['lo_quart'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['up_quart'].append(np.nan)

    return lwc_cdf_dict

def get_lwc_tot(lwc_vars, rho_air):

    lwc_tot = np.zeros(np.shape(rho_air))

    for i in range(1, 34):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...]
        lwc_tot += 4./3.*np.pi*rho_w*r3n_i/rho_air

    return lwc_tot

def make_lwc_cdf(case_label, lwc_cdf_dict):

    fig, ax = plt.subplots()

    colors = [magma_red, magma_pink, magma_orange]

    for i, lwc_cutoff_val in enumerate(lwc_cutoff_vals):
        color = colors[i]
        lwc_cutoff_key = str(lwc_cutoff_val)
        mean = np.array(lwc_cdf_dict[lwc_cutoff_key]['mean'])
        std = np.array(lwc_cdf_dict[lwc_cutoff_key]['std'])
        ax.fill_between(bin_radii*1.e6, mean - std, mean + std, \
                        color=color, alpha=0.5)
        ax.plot(bin_radii*1.e6, mean, color=color, \
                label=r'LWC$_{min}$='+lwc_cutoff_key)

    ax.set_xlim([1.e-1, 1.e4])
    ax.set_xlabel(r'r$_{max}$ ($\mu$m)')
    ax.set_ylabel(r'% LWC$_{tot}$')
    ax.set_xscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'lwc_cdf_mean_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

    fig, ax = plt.subplots()

    for i, lwc_cutoff_val in enumerate(lwc_cutoff_vals):
        color = colors[i]
        lwc_cutoff_key = str(lwc_cutoff_val)
        median = np.array(lwc_cdf_dict[lwc_cutoff_key]['mean'])
        lo_quart = np.array(lwc_cdf_dict[lwc_cutoff_key]['lo_quart'])
        up_quart = np.array(lwc_cdf_dict[lwc_cutoff_key]['up_quart'])
        ax.fill_between(bin_radii*1.e6, median - lo_quart, median + up_quart, \
                        color=color, alpha=0.5)
        ax.plot(bin_radii*1.e6, median, color=color, \
                label=r'LWC$_{min}$='+lwc_cutoff_key)

    ax.set_xlim([1.e-1, 1.e4])
    ax.set_xlabel(r'r$_{max}$ ($\mu$m)')
    ax.set_ylabel(r'% LWC$_{tot}$')
    ax.set_xscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'lwc_cdf_median_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
