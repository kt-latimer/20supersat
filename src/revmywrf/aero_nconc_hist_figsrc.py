"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

change_dsd_corr = False
cutoff_bins = False 
incl_rain = False
incl_vent = False

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_aero_nconc_hist(case_label, case_label_dict[case_label], \
                                        cutoff_bins, incl_rain, incl_vent)

def make_and_save_aero_nconc_hist(case_label, case_dir_name, \
                                        cutoff_bins, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    temp = met_vars['temp'][...]

    #close met file for memory
    met_file.close()
    dsdsum_file.close()

    filter_inds = np.logical_and.reduce((
                    (lwc < lwc_filter_val), \
                    (temp > 273)))

    #del vars for memory
    del lwc, temp

    asdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_asdsum_vars', 'r')
    asdsum_vars = asdsum_file.variables
    aero_nconc = asdsum_vars['nconc_sum'][...][filter_inds]

    #close file for memory
    asdsum_file.close()

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(aero_nconc, bins=30, density=False)
    ax.set_xlabel('nconc (m^-3)')
    ax.set_ylabel('Count')
    ax.set_title(case_label + ' Aerosol number concentration distribution' \
                    + ', cutoff_bins=' + str(cutoff_bins) \
                    + ', incl_rain=' + str(incl_rain) \
                    + ', incl_vent=' + str(incl_vent))
    outfile = FIG_DIR + versionstr + 'aero_nconc_hist_' \
            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
