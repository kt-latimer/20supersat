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
from halo.ss_functions import get_nconc_contribution_from_spectrum_var, \
                                get_meanr_contribution_from_spectrum_var, \
                                get_full_spectrum_bin_radii, \
                                get_full_spectrum_dlogDp, \
                                get_full_spectrum_dict, get_lwc_vs_t
from halo.utils import linregress
from wrf import DATA_DIR as WRF_DATA_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'halo': colors_arr[2], 'wrf_poll': colors_arr[5], \
                                    'wrf_unpoll': colors_arr[8]}
linestyles_dict = {'v1_': '-', 'v2_': '--'}
legends_dict = {'v1_': '1000-1500 m', 'v2_': '3000-4000 m'}
#versionstrs = ['v1_', 'v2_']
versionstrs = ['v1_']
                            
lwc_cutoff_val = 1.e-4
w_cutoff_val = 1

rmax = 102.e-6

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
case_color_key_dict = {'Polluted': 'wrf_poll', \
                        'Unpolluted': 'wrf_unpoll'}

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = True 

WRF_bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
WRF_bin_radii = WRF_bin_diams/2. 
WRF_log_bin_widths = np.array([np.log10(2.**(1./3.)) for i in range(33)])
HALO_bin_radii = get_full_spectrum_bin_radii(CAS_bins, CIP_bins, 'log')
HALO_log_bin_widths = get_full_spectrum_dlogDp(CAS_bins, CIP_bins)

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    alldates_spectrum_vent_dsd_dict = {}
    for i in range(1, 27):
        key = 'nconc_' + str(i)
        if change_cas_corr:
            key += '_corr'
        alldates_spectrum_vent_dsd_dict[key] = np.array([])

    for date in good_dates:
        alldates_spectrum_vent_dsd_dict = \
            update_vent_dsd_dict(alldates_spectrum_vent_dsd_dict, date)

    for i, key in enumerate(alldates_spectrum_vent_dsd_dict.keys()):
        inds = np.logical_not(np.isinf(alldates_spectrum_vent_dsd_dict[key]))
        print(HALO_bin_radii[i], np.nanmean(alldates_spectrum_vent_dsd_dict[key][inds]))

def update_vent_dsd_dict(alldates_spectrum_vent_dsd_dict, date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    spectrum_dict = get_full_spectrum_dict(cas_dict, \
                                cip_dict, change_cas_corr)

    lwc = get_lwc_vs_t(adlr_dict, spectrum_dict, cutoff_bins, rmax)
    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (w > w_cutoff_val)))

    #for j in range(1, 27):
    print(spectrum_dict['data'].keys())
    for var_name in spectrum_dict.keys():
        #var_name = 'nconc_' + str(j)
        #if change_cas_corr:
        #    var_name += '_corr'
        meanfr = get_meanr_contribution_from_spectrum_var(var_name, adlr_dict, \
            spectrum_dict, cutoff_bins, incl_rain, incl_vent, HALO_bin_radii)
        #print(j, date, np.nanmean(meanfr))
        alldates_spectrum_vent_dsd_dict[var_name] = np.concatenate(( \
                    alldates_spectrum_vent_dsd_dict[var_name], meanfr))

    return alldates_spectrum_vent_dsd_dict

def make_vent_dsd_fig_with_wrf_for_date(date, spectrum_vent_dsd_dict, case_label):

    fig, ax = plt.subplots()

    #print(HALO_log_bin_widths)
    #print(spectrum_vent_dsd_dict['mean']/HALO_log_bin_widths)
    print('halo', np.sum(spectrum_vent_dsd_dict['mean']))
    ax.plot(HALO_bin_radii*1.e6, \
            spectrum_vent_dsd_dict['mean']/HALO_log_bin_widths, \
            color=colors_dict['halo'], label='HALO')
    for versionstr in versionstrs:
        wrf_dsd_dict, wrf_vent_dsd_dict = get_wrf_dsd_dicts(case_label, versionstr)
        #print(versionstr, case_label, \
        #    np.sum(wrf_vent_dsd_dict['mean']*WRF_bin_radii))
        ax.plot(WRF_bin_radii*1.e6,
                wrf_vent_dsd_dict['data']*WRF_bin_radii/WRF_log_bin_widths, \
                color=colors_dict[case_color_key_dict[case_label]], \
                linestyle=linestyles_dict[versionstr], \
                label='WRF ' + legends_dict[versionstr])

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(r'$\frac{d(r \cdot f(r) \cdot N(r))}{d\log r}$ (cm$^{-3}$)')

    #ax.set_ylim([1.e-5, 1.e8])

    ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.legend()
    ax.set_title('HALO vs ' + case_label + ' WRF simulation')
    
    outfile = FIG_DIR + 'meanr_dsd_' + date + '_' + \
                            case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_wrf_dsd_dicts(case_label, versionstr):

    print(case_label)

    case_dsd_filename = WRF_DATA_DIR + versionstr + 'dsd_dict_slice_' + case_label + '_data.npy'
    #case_vent_dsd_filename = WRF_DATA_DIR + versionstr + 'vent_dsd_dict_slice_' \
    #                            + case_label + '_data.npy'
    case_vent_dsd_filename = WRF_DATA_DIR + versionstr + 'finiri_avg_' \
                                + case_label + '_data.npy'

    dsd_dict = np.load(case_dsd_filename, allow_pickle=True).item()
    vent_dsd_dict = np.load(case_vent_dsd_filename, allow_pickle=True).item()

    return dsd_dict, vent_dsd_dict

if __name__ == "__main__":
    main()
