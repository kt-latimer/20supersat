"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.met_data_functions import get_nconc_cloud, get_nconc_rain, \
                                                    get_temp, get_w
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
                            
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_nconc_hists(case_label, case_label_dict[case_label])
                                    
def make_and_save_nconc_hists(case_label, case_dir_name):

    #get input file variables
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    #get relevant physical qtys
    nconc = get_nconc_cloud(input_vars) + get_nconc_rain(input_vars) 
    temp = get_temp(input_vars) 
    w = get_w(input_vars) 

    #close files for memory
    input_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (w > w_cutoff), \
                    (temp > 273)))

    nconc = nconc[filter_inds]

    del filter_inds, temp, w #for memory

    fig, ax = plt.subplots()

    ax.hist(nconc, bins=40, density=False)

    ax.set_xlabel(r'nconc $\frac{1}{m^3}$')
    ax.set_ylabel(r'count')

    outfile = FIG_DIR + 'nconc_simple_hist_' + case_label + '_figure.png'
    plt.savefig(outfile)

    ax.set_yscale('log')

    outfile = FIG_DIR + 'nconc_simple_hist_log_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
