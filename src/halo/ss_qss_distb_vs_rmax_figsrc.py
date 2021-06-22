"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_ss_vs_t_cas, \
                            get_spliced_cas_and_cip_dicts, \
                            get_spliced_cas_and_cip_bins, \
                            get_nconc_contribution_from_cas_var, \
                            get_nconc_contribution_from_cip_var

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[2]
magma_pink = colors_arr[5]
magma_orange = colors_arr[8]

splice_methods = ['cas_over_cip', 'cip_over_cas', 'wt_avg']

lwc_filter_val = 1.e-4 #10**(-3.5)
w_cutoff = 1

z_min = -100
z_max = 6500

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

##
## physical constants
##
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

def main():
    
    for splice_method in splice_methods:
        make_ss_hist_series(splice_method)

def make_ss_hist_series(splice_method):

    rmax_vals = get_rmax_vals(splice_method)

    for rmax in rmax_vals:
        make_ss_hist_from_rmax(rmax, splice_method)

def get_rmax_vals(splice_method):

    cas_bin_radii, cip_bin_radii = get_spliced_cas_and_cip_bins(splice_method)

    return np.concatenate((cas_bin_radii, cip_bin_radii))

def make_ss_hist_from_rmax(rmax, splice_method):

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_qss_alldates = np.array([])

    for date in good_dates:
        ss_qss = get_ss_qss_data(date, rmax, splice_method)
        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)

    make_and_save_ss_qss_hist(ss_qss_alldates, rmax, splice_method)

def add_to_alldates_array(arr, arr_alldates):

    if arr_alldates is None:
        return arr
    else:
        return np.concatenate((arr_alldates, arr))

def get_ss_qss_data(date, rmax, splice_method):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    cas_dict, cip_dict = get_spliced_cas_and_cip_dicts(cas_dict, \
                        cip_dict, splice_method, change_cas_corr)

    lwc = get_lwc_with_rmax(adlr_dict, cas_dict, cip_dict, \
            change_cas_corr, False, rmax, splice_method)
    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']
    z = adlr_dict['data']['alt']
    ss_qss = get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, full_ss, \
                incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    if np.sum(filter_inds) != 0:
        ss_qss = ss_qss[filter_inds]
    else:
        ss_qss = np.array([])

    return ss_qss

def make_and_save_ss_qss_hist(ss_qss, rmax, splice_method):

    fig, ax = plt.subplots()

    if np.shape(ss_qss)[0] > 1:
        ax.hist(ss_qss, bins=30, density=False, color=magma_pink)
    ax.text(0.7, 0.8, r'$r_{max}$ = ' + str(rmax), \
            horizontalalignment='center', \
            verticalalignment='center', \
            transform = ax.transAxes)
    ax.set_xlabel(r'$SS_{pred}$ (%)')
    ax.set_ylabel(r'$N_{points}$')
    ax.set_title(splice_method)

    outfile = FIG_DIR + 'ss_qss_distb_' + str(rmax) + '_' + \
                                splice_method + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_lwc_with_rmax(adlr_dict, cas_dict, cip_dict, change_cas_corr, \
                                    cutoff_bins, rmax, splice_method):

    CAS_bin_radii, CIP_bin_radii = get_spliced_cas_and_cip_bins(splice_method)

    lwc = np.zeros(np.shape(adlr_dict['data']['time']))
    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    for i, r in enumerate(CAS_bin_radii):
        if r <= rmax:
            var_name = 'nconc_' + str(i+5)
            if change_cas_corr:
                var_name += '_corr'
            if var_name in cas_dict['data']:
                nconc_i = get_nconc_contribution_from_cas_var(var_name, adlr_dict, \
                    cas_dict, change_cas_corr, cutoff_bins, incl_rain, incl_vent)
                lwc += nconc_i*r**3.*4./3.*np.pi*rho_w/rho_air

    for i, r in enumerate(CIP_bin_radii):
        if r <= rmax:
            var_name = 'nconc_' + str(i+1)
            nconc_i = get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
                    cip_dict)
            lwc += nconc_i*r**3.*4./3.*np.pi*rho_w/rho_air

    return lwc

if __name__ == "__main__":
    main()
