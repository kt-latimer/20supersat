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
        make_and_save_ss_qss_vs_ss_wrf(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)

def make_and_save_ss_qss_vs_ss_wrf(case_label, case_dir_name, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = met_vars['LWC_cloud'][...] + met_vars['LWC_rain'][...]
    #lwc = get_lwc(met_vars, dsdsum_vars, False, True, False)
    nconc = met_vars['nconc_cloud'][...] + met_vars['nconc_rain'][...]

    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (nconc < 1.e7), \
                    (w > w_cutoff), \
                    (temp > 273)))

    nconc = nconc[filter_inds]
    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

    m, b, R, sig = linregress(ss_qss, ss_wrf)
    print(case_label)
    print(m, b, R**2)
    N_points = np.sum(ss_qss < 200)
    print('# pts total: ' + str(N_points))
    print('max: ' + str(np.nanmax(ss_qss)))
    print('# pts ss > 2%: ' + str(np.sum(ss_qss > 2)))
   
    print_point_count_per_quadrant(ss_qss, ss_wrf)

    fig, ax = plt.subplots()
    
    ss_bins = get_ss_bins(ss_min, ss_max, d_ss)
    nconcs = get_nconcs(nconc, ss_bins, ss_qss, ss_wrf)

    im = ax.pcolormesh(ss_bins, ss_bins, nconcs, cmap=plt.cm.magma_r)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'Total num conc ($\frac{1}{m^3}$)')

    ax.set_aspect('equal', 'box')

    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'$SS_{WRF}$ (%)')
    fig.suptitle('Actual versus approximated supersaturation - WRF ' + case_label)

    outfile = FIG_DIR + 'lo_nconc_heatmap_ss_qss_vs_ss_wrf_' \
                            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

def get_nconcs(nconc, ss_bins, ss_qss, ss_wrf):

    n_bins = np.shape(ss_bins)[0] - 1
    nconcs = np.zeros((n_bins, n_bins))

    for i, val1 in enumerate(ss_bins[:-1]):
        for j, val2 in enumerate(ss_bins[:-1]):
            ss_qss_lo = val1
            ss_qss_hi = ss_bins[i+1]
            ss_wrf_lo = val2
            ss_wrf_hi = ss_bins[j+1]

            bin_inds = np.logical_and.reduce((( \
                            (ss_qss >= ss_qss_lo), \
                            (ss_qss < ss_qss_hi), \
                            (ss_wrf >= ss_wrf_lo), \
                            (ss_wrf < ss_wrf_hi))))

            if np.sum(bin_inds) > 0:
                nconc_filt = nconc[bin_inds]
                nconcs[i, j] = np.mean(nconc_filt)
            else:
                nconcs[i, j] = np.nan 

    return nconcs

def get_ss_bins(ss_min, ss_max, d_ss):

    ss_bins = np.arange(ss_min, ss_max, d_ss)

    return ss_bins

def print_point_count_per_quadrant(ss_qss, ss_wrf):

    n_q1 = np.sum(np.logical_and(ss_qss > 0, ss_wrf > 0))
    n_q2 = np.sum(np.logical_and(ss_qss < 0, ss_wrf > 0))
    n_q3 = np.sum(np.logical_and(ss_qss < 0, ss_wrf < 0))
    n_q4 = np.sum(np.logical_and(ss_qss > 0, ss_wrf < 0))
    
    print('Number of points in Q1:', n_q1)
    print('Number of points in Q2:', n_q2)
    print('Number of points in Q3:', n_q3)
    print('Number of points in Q4:', n_q4)
    print()
    print('Domain:', np.min(ss_qss), np.max(ss_qss))
    print('Range:', np.min(ss_wrf), np.max(ss_wrf))

if __name__ == "__main__":
    main()
