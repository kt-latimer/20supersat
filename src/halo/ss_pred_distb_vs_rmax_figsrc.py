"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.ss_functions import get_spliced_cas_and_cip_dicts, get_ss_vs_t_cas, \
    get_nconc_contribution_from_cas_var, get_nconc_contribution_from_cip_var, \
    get_meanr_contribution_from_cas_var, get_meanr_contribution_from_cip_var

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

CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)
CIP_bin_radii = np.sqrt((CIP_bins['upper']*CIP_bins['lower'])/4.)

def main():

    ss_max_vals = []
    rmax_vals = np.concatenate((CAS_bin_radii, CIP_bin_radii[1:])) 
    
    for rmax in rmax_vals:
        ss_vals = make_ss_hist(rmax)
        print(rmax)
        print(np.shape(ss_vals))
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

    outfile = FIG_DIR + 'ss_pred_vs_rmax_' + str(rmax) + '_v2_figure.png'
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

    ss_pred = get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, full_ss, \
                incl_rain, incl_vent, 'cas_over_cip')
    cas_dict, cip_dict = get_spliced_cas_and_cip_dicts(cas_dict, \
                                        cip_dict, 'cas_over_cip')
    lwc = get_lwc_with_rmax(adlr_dict, cas_dict, cip_dict, rmax)
    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    return ss_pred[filter_inds]

def get_lwc_with_rmax(adlr_dict, cas_dict, cip_dict, rmax):

    lwc = np.zeros(np.shape(adlr_dict['data']['time']))
    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    for i, r in enumerate(CAS_bin_radii):
        if r <= rmax:
            var_name = 'nconc_' + str(i+5)
            if change_cas_corr:
                var_name += '_corr'
            #nconc_i = get_nconc_contribution_from_cas_var(var_name, adlr_dict, \
            #    cas_dict, change_cas_corr, cutoff_bins, incl_rain)
            meanr_i = get_meanr_contribution_from_cas_var(var_name, adlr_dict, \
                cas_dict, change_cas_corr, cutoff_bins, incl_rain, False)
            #lwc += nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air
            lwc += meanr_i*r**2.*4./3.*np.pi*rho_l/rho_air

    for i, r in enumerate(CIP_bin_radii):
        if r <= rmax:
            var_name = 'nconc_' + str(i+1)
            #nconc_i = get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
            #        cip_dict)
            meanr_i = get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                    cip_dict, incl_rain, False)
            #lwc += nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air
            lwc += meanr_i*r**2.*4./3.*np.pi*rho_l/rho_air

    return lwc

if __name__ == "__main__":
    main()
