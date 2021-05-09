"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.ss_functions import get_nconc_contribution_from_cas_var, \
                                get_meanr_contribution_from_cas_var, \
                                get_nconc_contribution_from_cip_var, \
                                get_meanr_contribution_from_cip_var, \
                                get_spliced_cas_and_cip_dicts
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[1]
magma_pink = colors_arr[5]
magma_orange = colors_arr[9]

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = False 

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of water (kg/m^3) 

CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)
CIP_bin_radii = np.sqrt((CIP_bins['upper']*CIP_bins['lower'])/4.)

rmax_vals = np.concatenate((CAS_bin_radii, CIP_bin_radii[1:])) 
lwc_cutoff_vals = [-1, 1.e-5, 1.e-4]

def main():

    lwc_cdf_dict = get_lwc_cdf_dict()
    make_lwc_cdf(lwc_cdf_dict)

def get_lwc_cdf_dict():

    tot_adlr_dict, tot_cas_dict, tot_cip_dict = get_dicts_from_all_dates()
    tot_cas_dict, tot_cip_dict = get_spliced_cas_and_cip_dicts( \
                        tot_cas_dict, tot_cip_dict, 'cas_over_cip')
    lwc_tot = get_lwc_tot(tot_adlr_dict, tot_cas_dict, tot_cip_dict)
    lwc = np.zeros(np.shape(tot_adlr_dict['data']['time']))
    lwc_cdf_dict = {}

    for rmax in rmax_vals:
        lwc += get_lwc_contribution(rmax, tot_adlr_dict, \
                                tot_cas_dict, tot_cip_dict)
        for lwc_cutoff_val in lwc_cutoff_vals:
            lwc_cdf_dict = update_lwc_cdf_dict(lwc, lwc_cdf_dict, \
                                            lwc_cutoff_val, lwc_tot)

    return lwc_cdf_dict
             
def update_lwc_cdf_dict(lwc, lwc_cdf_dict, lwc_cutoff_val, lwc_tot):

    filter_inds = lwc > lwc_cutoff_val
    lwc_cutoff_key = str(lwc_cutoff_val)

    if lwc_cutoff_key not in lwc_cdf_dict.keys():
        lwc_cdf_dict[lwc_cutoff_key] = {'mean': [], 'std': [], \
                    'median': [], 'lo_quart': [], 'up_quart': []}

    if np.sum(filter_inds) != 0:
        lwc_cdf_dict[lwc_cutoff_key]['mean'].append( \
            np.nanmean(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['std'].append( \
            np.nanstd(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['median'].append( \
            np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds]))
        lwc_cdf_dict[lwc_cutoff_key]['lo_quart'].append( \
            (np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds]) - \
            np.percentile(lwc[filter_inds]/lwc_tot[filter_inds], 25)))
        lwc_cdf_dict[lwc_cutoff_key]['up_quart'].append( \
            (np.percentile(lwc[filter_inds]/lwc_tot[filter_inds], 75) - \
            np.nanmedian(lwc[filter_inds]/lwc_tot[filter_inds])))
    else:
        lwc_cdf_dict[lwc_cutoff_key]['mean'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['std'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['median'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['lo_quart'].append(np.nan)
        lwc_cdf_dict[lwc_cutoff_key]['up_quart'].append(np.nan)

    return lwc_cdf_dict

def get_dicts_from_all_dates():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    tot_adlr_dict, tot_cas_dict, tot_cip_dict = \
            get_dicts_from_one_date(good_dates[0])

    for date in good_dates[1:]:
        adlr_dict, cas_dict, cip_dict = get_dicts_from_one_date(date)
        tot_adlr_dict = update_tot_dict(tot_adlr_dict, adlr_dict)
        tot_cas_dict = update_tot_dict(tot_cas_dict, cas_dict)
        tot_cip_dict = update_tot_dict(tot_cip_dict, cip_dict)

    return tot_adlr_dict, tot_cas_dict, tot_cip_dict

def get_dicts_from_one_date(date):
    
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 273), \
                            (w > 1)))
    
    adlr_dict = get_filtered_data(adlr_dict, filter_inds) 
    cas_dict = get_filtered_data(cas_dict, filter_inds) 
    cip_dict = get_filtered_data(cip_dict, filter_inds) 

    return adlr_dict, cas_dict, cip_dict

def update_tot_dict(tot_data_dict, data_dict):

    keys = tot_data_dict['data'].keys()

    for key in keys:
        tot_data_dict['data'][key] = np.concatenate( \
            (tot_data_dict['data'][key], data_dict['data'][key]))

    return tot_data_dict

def get_filtered_data(data_dict, filter_inds):

    keys = data_dict['data'].keys()

    for key in keys:
        data_dict['data'][key] = data_dict['data'][key][filter_inds]

    return data_dict

def get_lwc_tot(tot_adlr_dict, tot_cas_dict, tot_cip_dict):

    lwc_tot = np.zeros(np.shape(tot_adlr_dict['data']['time']))
    rho_air = tot_adlr_dict['data']['pres']/(R_a*tot_adlr_dict['data']['temp'])

    for i, r in enumerate(CAS_bin_radii):
        var_name = 'nconc_' + str(i+5)
        if change_cas_corr:
            var_name += '_corr'
        nconc_i = get_nconc_contribution_from_cas_var(var_name, \
                tot_adlr_dict, tot_cas_dict, change_cas_corr, \
                cutoff_bins, incl_rain, incl_vent)
        lwc_tot += nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air

    for i, r in enumerate(CIP_bin_radii):
        var_name = 'nconc_' + str(i+1)
        nconc_i = get_nconc_contribution_from_cip_var(var_name, \
                                        tot_adlr_dict, tot_cip_dict)
        lwc_tot += nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air

    return lwc_tot

def get_lwc_contribution(rmax, tot_adlr_dict, tot_cas_dict, tot_cip_dict):

    rho_air = tot_adlr_dict['data']['pres']/(R_a*tot_adlr_dict['data']['temp'])

    for i, r in enumerate(CAS_bin_radii):
        if r == rmax:
            var_name = 'nconc_' + str(i+5)
            if change_cas_corr:
                var_name += '_corr'
            nconc_i = get_nconc_contribution_from_cas_var(var_name, \
                    tot_adlr_dict, tot_cas_dict, change_cas_corr, \
                    cutoff_bins, incl_rain, incl_vent)
            lwc_contrib = nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air
            return lwc_contrib

    for i, r in enumerate(CIP_bin_radii):
        if r == rmax:
            var_name = 'nconc_' + str(i+1)
            nconc_i = get_nconc_contribution_from_cip_var(var_name, \
                                            tot_adlr_dict, tot_cip_dict)
            lwc_contrib = nconc_i*r**3.*4./3.*np.pi*rho_l/rho_air
            return lwc_contrib

def make_lwc_cdf(lwc_cdf_dict):

    fig, ax = plt.subplots()

    colors = [magma_red, magma_pink, magma_orange]

    for i, lwc_cutoff_val in enumerate(lwc_cutoff_vals):
        color = colors[i]
        lwc_cutoff_key = str(lwc_cutoff_val)
        mean = np.array(lwc_cdf_dict[lwc_cutoff_key]['mean'])
        std = np.array(lwc_cdf_dict[lwc_cutoff_key]['std'])
        ax.fill_between(rmax_vals*1.e6, mean - std, mean + std, \
                        color=color, alpha=0.5)
        ax.plot(rmax_vals*1.e6, mean, color=color, \
                label=r'LWC$_{min}$='+lwc_cutoff_key)

    ax.set_xlim([1.e-1, 1.e4])
    ax.set_xlabel(r'r$_{max}$ ($\mu$m)')
    ax.set_ylabel(r'% LWC$_{tot}$')
    ax.set_xscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'alldates_lwc_cdf_mean_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

    fig, ax = plt.subplots()

    for i, lwc_cutoff_val in enumerate(lwc_cutoff_vals):
        color = colors[i]
        lwc_cutoff_key = str(lwc_cutoff_val)
        median = np.array(lwc_cdf_dict[lwc_cutoff_key]['mean'])
        lo_quart = np.array(lwc_cdf_dict[lwc_cutoff_key]['lo_quart'])
        up_quart = np.array(lwc_cdf_dict[lwc_cutoff_key]['up_quart'])
        ax.fill_between(rmax_vals*1.e6, median - lo_quart, median + up_quart, \
                        color=color, alpha=0.5)
        ax.plot(rmax_vals*1.e6, median, color=color, \
                label=r'LWC$_{min}$='+lwc_cutoff_key)

    ax.set_xlim([1.e-1, 1.e4])
    ax.set_xlabel(r'r$_{max}$ ($\mu$m)')
    ax.set_ylabel(r'% LWC$_{tot}$')
    ax.set_xscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'alldates_lwc_cdf_median_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
