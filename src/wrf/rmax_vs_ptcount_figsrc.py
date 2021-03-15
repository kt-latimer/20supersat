"""
make smaller data files for quantities after applying LWC, vertical wind
velocity, and temperature filters
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

###
### bin sizes and regime params
###
bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

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
    lwc_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_lwc_spectrum_vars', 'r')
    lwc_vars = lwc_file.variables

    #get relevant physical qtys
    rho_air = met_vars['rho_air'][...]
    ss_wrf = met_vars['ss_wrf'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    filter_inds = np.logical_and.reduce((
                            (temp > 273), \
                            (w > 1)))

    del temp, w #for memory

    rho_air = rho_air[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    lwc_dict = {}

    for i in range(1, 34):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...][filter_inds]
        lwc_i = 4./3.*np.pi*rho_w*r3n_i/rho_air
        lwc_dict[var_name] = lwc_i

    del filter_inds, lwc_i, r3n_i, rho_air #for memory

    #close file for memory
    met_file.close()
    
    fig, [ax0, ax1, ax2] = plt.subplots(1, 3)

    lwc_cutoff_vals = [-5, -4.75, -4.5, -4.25, -4]

    print(bin_radii*1.e6)
    for val in lwc_cutoff_vals:
        lwc_cutoff_val = 10**val
        print(val)
        n_tot, n_q4 = get_n_pts(lwc_dict, lwc_cutoff_val, ss_wrf)
        print(n_tot)
        print(n_q4)
        print(n_q4/n_tot)
        ax0.scatter(bin_radii*1.e6, n_tot, \
            label=r'log$_{10}$(LWC$_{min}$)='+str(val))
        ax1.scatter(bin_radii*1.e6, n_q4, \
            label=r'log$_{10}$(LWC$_{min}$)='+str(val))
        ax2.scatter(bin_radii*1.e6, n_q4/n_tot, \
            label=r'log$_{10}$(LWC$_{min}$)='+str(val))

    ax0.set_xscale('log')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax0.set_xlabel(r'r ($\mu$m)')
    ax1.set_xlabel(r'r ($\mu$m)')
    ax2.set_xlabel(r'r ($\mu$m)')
    ax2.legend()

    outfile = FIG_DIR + 'rmax_vs_ptcount_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_n_pts(lwc_dict, lwc_cutoff_val, ss_wrf):

    n_tot_arr = []
    n_q4_arr = []

    for i in range(1, 34):
        var_key = 'r3n_sum_' + str(i)
        lwc_i = lwc_dict[var_key]

        filter_inds_i = lwc_i > lwc_cutoff_val
        n_tot = np.sum(filter_inds_i)

        q4_filter_inds_i = ss_wrf[filter_inds_i] < 0
        n_q4 = np.sum(q4_filter_inds_i)

        n_tot_arr.append(n_tot)
        n_q4_arr.append(n_q4)

    return np.array(n_tot_arr), np.array(n_q4_arr)

if __name__ == "__main__":
    main()
