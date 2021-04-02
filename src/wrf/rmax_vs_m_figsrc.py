"""
make smaller data files for quantities after applying LWC, vertical wind
velocity, and temperature filters
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_ss_qss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

###
### bin sizes and regime params
###
bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

rho_w = 1000. #density of water (kg/m^3)

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_rmax_vs_m(case_label) 

def make_and_save_rmax_vs_m(case_label):

    print(case_label)
    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get lwc file variables
    lwc_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_lwc_spectrum_vars', 'r')
    lwc_vars = lwc_file.variables

    #get relevant physical qtys
    rho_air = met_vars['rho_air'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    filter_inds = np.logical_and.reduce((
                            (temp > 273), \
                            (w > 1)))

    del temp, w #for memory

    rho_air = rho_air[filter_inds]
    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    lwc_dict = {}

    for i in range(1, 34):
        var_name = 'r3n_sum_' + str(i)
        r3n_i = lwc_vars[var_name][...][filter_inds]
        lwc_i = 4./3.*np.pi*rho_w*r3n_i/rho_air
        lwc_dict[var_name] = lwc_i

    del lwc_i, r3n_i, rho_air #for memory

    #close file for memory
    met_file.close()
    
    fig, ax = plt.subplots()

    lwc_cutoff_vals = [-5, -4.75, -4.5, -4.25, -4]

    for val in lwc_cutoff_vals:
        lwc_cutoff_val = 10**val
        m_vals = get_m_vals(lwc_dict, lwc_cutoff_val, ss_qss, ss_wrf)
        ax.scatter(bin_radii*1.e6, m_vals, \
            label=r'log$_{10}$(LWC$_{min}$)='+str(val))

    ax.set_xscale('log')
    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(r'LSR slope')
    ax.legend()

    outfile = FIG_DIR + 'rmax_vs_m_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_m_vals(lwc_dict, lwc_cutoff_val, ss_qss, ss_wrf):

    m_vals = []

    for i in range(1, 34):
        var_key = 'r3n_sum_' + str(i)
        lwc_i = lwc_dict[var_key]

        filter_inds_i = lwc_i > lwc_cutoff_val

        if np.sum(filter_inds_i) > 0:
            ss_qss_i = ss_qss[filter_inds_i]
            ss_wrf_i = ss_wrf[filter_inds_i]
            m, b, R, sig = linregress(ss_qss_i, ss_wrf_i)
            m_vals.append(m)
        else:
            m_vals.append(np.nan)

    return np.array(m_vals)

if __name__ == "__main__":
    main()
