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
                                get_meanr_contribution_from_cip_var
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]
magma_orange = colors_arr[1]

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = False 

CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)
CIP_bin_radii = np.sqrt((CIP_bins['upper']*CIP_bins['lower'])/4.)
CAS_upper_bin_radii = CAS_bins['upper']/2.
CIP_upper_bin_radii = CIP_bins['upper']/2.
CAS_lower_bin_radii = CAS_bins['lower']/2.
CIP_lower_bin_radii = CIP_bins['lower']/2.
CAS_log_bin_widths = np.log10(CAS_bins['upper']/CAS_bins['lower'])
CIP_log_bin_widths = np.log10(CIP_bins['upper']/CIP_bins['lower'])

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    alldates_cas_vent_dsd_dict = {}
    alldates_cip_vent_dsd_dict = {}
    n_pts_tot = 0

    for date in good_dates:
        alldates_cas_vent_dsd_dict, alldates_cip_vent_dsd_dict = \
            get_vent_dsd_dicts(date, alldates_cas_vent_dsd_dict, \
                                        alldates_cip_vent_dsd_dict)

    cas_mean, cas_std = get_mean_and_std(alldates_cas_vent_dsd_dict, \
                                                        CAS_bin_radii)
    cip_mean, cip_std = get_mean_and_std(alldates_cip_vent_dsd_dict, \
                                                        CIP_bin_radii)

    print(cas_mean)
    make_vent_dsd_fig_for_date('alldates', cas_mean, cas_std, cip_mean, cip_std)

def get_vent_dsd_dicts(date, alldates_cas_vent_dsd_dict, \
                                alldates_cip_vent_dsd_dict):

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

    for j in range(5, 17):
        var_name = 'nconc_' + str(j)
        if change_cas_corr:
            var_name += '_corr'
        if not (var_name in alldates_cas_vent_dsd_dict.keys()):
            alldates_cas_vent_dsd_dict[var_name] = np.array([])
        meanfr = get_meanr_contribution_from_cas_var(var_name, adlr_dict, \
                cas_dict, change_cas_corr, cutoff_bins, incl_rain, incl_vent)
        alldates_cas_vent_dsd_dict[var_name] = np.concatenate( \
                (meanfr, alldates_cas_vent_dsd_dict[var_name]))

    for j in range(1, 20):
        var_name = 'nconc_' + str(j)
        if not (var_name in alldates_cip_vent_dsd_dict.keys()):
            alldates_cip_vent_dsd_dict[var_name] = np.array([])
        meanfr = get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                cip_dict, incl_rain, incl_vent)
        alldates_cip_vent_dsd_dict[var_name] = np.concatenate( \
                (meanfr, alldates_cip_vent_dsd_dict[var_name]))

    return alldates_cas_vent_dsd_dict, alldates_cip_vent_dsd_dict

def get_mean_and_std(alldates_vent_dsd_dict, bin_radii):

    mean = []
    std = []

    for key in alldates_vent_dsd_dict.keys():
        meanr = alldates_vent_dsd_dict[key]
        not_inf_inds = np.logical_not(np.isinf(meanr))
        mean.append(np.nanmean(meanr[not_inf_inds]))
        std.append(np.nanstd(meanr[not_inf_inds]))

    mean = np.array(mean)*bin_radii**2.
    std = np.array(std)*bin_radii**2.

    return mean, std

def make_vent_dsd_fig_for_date(date, cas_mean, cas_std, cip_mean, cip_std):

    fig, ax = plt.subplots()
    
    ax.errorbar(CAS_bin_radii*1.e6, \
                cas_mean/CAS_log_bin_widths, \
                yerr=cas_std/CAS_log_bin_widths, \
                color='grey', alpha=0.3, fmt='o')
    ax.step(CAS_lower_bin_radii*1.e6, \
            cas_mean/CAS_log_bin_widths, \
            where='post', color=magma_pink)
    ax.plot([CAS_lower_bin_radii[-1]*1.e6, CAS_upper_bin_radii[-1]*1.e6], \
            [cas_mean[-1]/CAS_log_bin_widths[-1], \
            cas_mean[-1]/CAS_log_bin_widths[-1]], \
            color=magma_pink, label='CAS')
    ax.errorbar(CIP_bin_radii*1.e6, \
                cip_mean/CIP_log_bin_widths, \
                yerr=cip_std/CIP_log_bin_widths, \
                color='grey', alpha=0.3, fmt='o')
    ax.step(CIP_lower_bin_radii*1.e6, \
            cip_mean/CIP_log_bin_widths, \
            where='post', color=magma_orange)
    ax.plot([CIP_lower_bin_radii[-1]*1.e6, CIP_upper_bin_radii[-1]*1.e6], \
            [cip_mean[-1]/CIP_log_bin_widths[-1], \
            cip_mean[-1]/CIP_log_bin_widths[-1]], \
            color=magma_orange, label='CIP')

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(r'$\frac{d(r \cdot f(r) \cdot N(r))}{d\log r}$ (cm$^{-3}$)')

    #ax.set_ylim([1.e-5, 1.e7])
    ax.set_xlim([1.e-1, 1.e4])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend()
    
    outfile = FIG_DIR + 'alldates_meanr_dsd_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
