"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
simple form of ss_qss
"""
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():
    
    for case_label in case_label_dict.keys():
        print_avg_lwc_in_slice(case_label)

def print_avg_lwc_in_slice(case_label):

    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get relevant physical qtys
    lwc = met_vars['LWC_cloud'][...] + met_vars['LWC_rain'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #close files for memory
    met_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (z > 1500), \
                    (z < 2500), \
                    (temp > 273)))

    lwc = lwc[filter_inds]
    print(case_label)
    print(np.mean(lwc))
    print(np.nanmean(lwc))

if __name__ == "__main__":
    main()
