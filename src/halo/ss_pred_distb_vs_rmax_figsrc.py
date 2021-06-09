"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.ss_functions import get_lwc_vs_t, get_ss_vs_t, \
                              get_full_spectrum_bin_radii, \
                              get_full_spectrum_dict

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

lwc_filter_val = 1.e-4
w_cutoff = 1

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = True
full_ss = True

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of water (kg/m^3) 

def main():

    ss_max_vals = []
    rmax_vals = get_full_spectrum_bin_radii(CAS_bins, CIP_bins, 'log') 
    print(rmax_vals)
    return
    
    for rmax in rmax_vals:
        ss_vals = make_ss_hist(rmax)
        if np.shape(ss_vals)[0] != 0:
            ss_max_vals.append(np.nanmax(ss_vals))

    fig, ax = plt.subplots()

    ax.scatter(rmax_vals, ss_max_vals)

    outfile = FIG_DIR + 'ss_pred_max_vs_rmax_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def make_ss_hist(rmax):

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_vals = np.array([])

    for date in good_dates:
        date_ss_vals = get_ss_vals(date, rmax)
        ss_vals = np.concatenate((ss_vals, date_ss_vals))

    ss_vals = ss_vals[np.logical_not(np.isnan(ss_vals))]

    if np.shape(ss_vals)[0] == 0:
        return ss_vals

    fig, ax = plt.subplots()

    ax.hist(ss_vals, bins=30, density=False)

    fig.suptitle(r'$r_{max}$ = ' + str(rmax))

    outfile = FIG_DIR + 'ss_pred_vs_rmax_' + str(rmax) + '_v3_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

    return ss_vals

def get_ss_vals(date, rmax):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(cas_dict, \
                                cip_dict, change_cas_corr)

    ss_pred = get_ss_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)
    lwc = get_lwc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, rmax)

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    return ss_pred[filter_inds]

if __name__ == "__main__":
    main()
