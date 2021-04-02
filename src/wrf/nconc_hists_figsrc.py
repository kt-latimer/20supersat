"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
                            
lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_nconc_hists(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)

def make_and_save_nconc_hists(case_label, case_dir_name, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get relevant physical qtys
    lwc = met_vars['LWC_cloud'][...] + met_vars['LWC_rain'][...]
    nconc = met_vars['nconc_rain'][...]
    #nconc = met_vars['nconc_cloud'][...] + met_vars['nconc_rain'][...]
    ss_wrf = met_vars['ss_wrf'][...]*100
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    #close files for memory
    met_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (nconc < 1.e6), \
                    (w > w_cutoff), \
                    (temp > 273)))

    nconc = nconc[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    q1_inds = ss_wrf > 0
    q4_inds = np.logical_not(q1_inds)

    nconc_q1 = nconc[q1_inds]
    nconc_q4 = nconc[q4_inds]

    del filter_inds, lwc, nconc, q1_inds, q4_inds, ss_wrf, temp, w

    fig, ax = plt.subplots()

    ax.hist(nconc_q1, bins=30, alpha=0.6)
    ax.hist(nconc_q4, bins=30, alpha=0.6)

    ax.set_xlabel(r'nconc $\frac{1}{m^3}$')
    ax.set_ylabel(r'count')

    outfile = FIG_DIR + 'nconc_hists_more_zoomed_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
