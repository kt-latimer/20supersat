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
    lh = met_vars['LH'][...]
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]

    #filtering to determine total condensational latent heating
    filter_inds = np.logical_and.reduce((
                    (lh > 0), \
                    (temp > 273)))

    lh = lh[filter_inds]

    lwc_dict = {}

    for i in range(1, 34):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...]
        lwc_i = 4./3.*np.pi*rho_w*r3n_i/rho_air
        lwc_dict[var_name] = lwc_i[filter_inds]

    del filter_inds, lwc_i, r3n_i, rho_air, temp #for memory

    #close file for memory
    met_file.close()
    
    fig, ax = plt.subplots()

    lwc_cutoff_vals = [-5, -4.75, -4.5, -4.25, -4]

    for val in lwc_cutoff_vals:
        lwc_cutoff_val = 10**val
        lh_cdf = get_lh_cdf(lh, lwc_dict, lwc_cutoff_val)
        ax.plot(bin_radii[:-1]*1.e6, lh_cdf, \
            label=r'log$_{10}$(LWC$_{min}$)='+str(val))

    ax.set_xscale('log')
    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel('(Positive) LH CDF')
    ax.legend()

    outfile = FIG_DIR + 'lwc_spectrum_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_lh_cdf(lh, lwc_dict, lwc_cutoff_val):

    lh_cdf = []
    lh_tot = np.sum(lh)

    for i in range(1, 33):
        var_key = 'r3n_sum_' + str(i)
        lwc_i = lwc_dict[var_key]

        filter_inds_i = lwc_i > lwc_cutoff_val

        lh_tot_i = np.sum(lh[filter_inds_i])
        lh_cdf.append(lh_tot_i/lh_tot)

    return np.array(lh_cdf)

if __name__ == "__main__":
    main()
