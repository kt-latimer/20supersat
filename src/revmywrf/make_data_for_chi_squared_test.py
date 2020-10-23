"""
make and save data dictionaries with ss_qss and altitude data.

all references herein to 'ss' are quasi-steady-state approximations to
supersaturation (for brevity)

also, 'exp' = 'experimental' and 'sim' = 'simulated'
"""
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
#versionstr = 'v1_' #cloud/rain diam bndy ~100 um (same as Qindan orig code)
versionstr = 'v2_' #cloud/rain diam bndy ~50 um and no rain (to compare to caipeex 2009)

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = True 
incl_rain = False
incl_vent = False
full_ss = True

def main():

    ss_sim_dict = {'Polluted': None, 'Unpolluted': None}
    z_sim_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        (ss_sim_dict, z_sim_dict) = update_dicts_for_chi_squared(case_label, \
                                                case_label_dict[case_label], \
                                                cutoff_bins, full_ss, \
                                                incl_rain, incl_vent, \
                                                ss_sim_dict, z_sim_dict)

    np.save(DATA_DIR + versionstr + 'ss_sim_dict.npy', ss_sim_dict)
    np.save(DATA_DIR + versionstr + 'z_sim_dict.npy', z_sim_dict)

def update_dicts_for_chi_squared(case_label, \
                case_dir_name, \
                cutoff_bins, full_ss, \
                incl_rain, incl_vent, \
                ss_sim_dict, z_sim_dict):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                #'wrfout_d01_all_dsdsum_vars_v2', 'r') #v1
                                'wrfout_d01_all_dsdsum_vars', 'r') #v2 
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

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    #(np.abs(w) > w_cutoff), \
                    (w > w_cutoff), \
                    (temp > 273)))
                    #(temp > -2730)))

    del lwc, w, temp #for memory

    ss_qss = ss_qss[filter_inds]
    z = z[filter_inds]

    del filter_inds #for memory

    print(case_label)
    ss_sim_dict[case_label] = ss_qss
    z_sim_dict[case_label] = z 

    del ss_qss, z #for memory

    return (ss_sim_dict, z_sim_dict)

if __name__ == "__main__":
    main()
