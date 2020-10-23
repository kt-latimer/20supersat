"""
make and save data dictionaries with ss_qss and altitude data.

all references herein to 'ss' are quasi-steady-state approximations to
supersaturation (for brevity)

also, 'exp' = 'experimental' and 'sim' = 'simulated'
"""
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_meanr, get_nconc, linregress

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

    meanr_sim_dict = {'Polluted': None, 'Unpolluted': None}
    nconc_sim_dict = {'Polluted': None, 'Unpolluted': None}
    w_sim_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        (meanr_sim_dict, nconc_sim_dict, w_sim_dict) = \
                update_dicts_for_chi_squared(case_label, \
                    case_label_dict[case_label], cutoff_bins, \
                    incl_rain, incl_vent, meanr_sim_dict, \
                    nconc_sim_dict, w_sim_dict)

    np.save(DATA_DIR + versionstr + 'meanr_sim_dict.npy', meanr_sim_dict)
    np.save(DATA_DIR + versionstr + 'nconc_sim_dict.npy', nconc_sim_dict)
    np.save(DATA_DIR + versionstr + 'w_sim_dict.npy', w_sim_dict)

def update_dicts_for_chi_squared(case_label, \
        case_dir_name, cutoff_bins, \
        incl_rain, incl_vent, meanr_sim_dict, \
        nconc_sim_dict, w_sim_dict):

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
    meanr = get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent)

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

    del lwc, temp #for memory

    meanr = meanr[filter_inds]
    nconc = nconc[filter_inds]
    w = w[filter_inds]

    del filter_inds #for memory

    print(case_label)
    meanr_sim_dict[case_label] = meanr
    nconc_sim_dict[case_label] = nconc 
    w_sim_dict[case_label] = w 

    del meanr, nconc, w #for memory

    return (meanr_sim_dict, nconc_sim_dict, w_sim_dict)

if __name__ == "__main__":
    main()
