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
linestyles_dict = {'v1_': '-', 'v2_': '--', 'v3_': '-'}
legends_dict = {'v1_': '1000-1500 m', 'v2_': '3000-4000 m', 'v3_': '1500-2500 m'}
#versionstrs = ['v1_', 'v2_']
versionstrs = ['v3_']
#versionstrs = ['v1_']
                            
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

    alldates_spectrum_vent_dict = None 

    for date in good_dates:
        print(date)
        #print(alldates_spectrum_vent_dict)
        date_spectrum_vent_dict = get_date_spectrum_vent_dict(date)
        if alldates_spectrum_vent_dict == None:
            print('hi')
            alldates_spectrum_vent_dict = date_spectrum_vent_dict
        else:
            alldates_spectrum_vent_dict = \
                update_alldates_spectrum_vent_dict( \
                        alldates_spectrum_vent_dict, \
                        date_spectrum_vent_dict)
            
    #for key in alldates_spectrum_vent_dict.keys():
    #    print(key)
    #    print(np.nanmean(alldates_spectrum_vent_dict[key]))
    for case_label in case_label_dict.keys():
        make_vent_dsd_fig_with_wrf_for_date('alldates', \
                alldates_spectrum_vent_dict, case_label)

def get_date_spectrum_vent_dict(date):

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
    z = adlr_dict['data']['alt']

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (z > 1500), \
                            (z < 2500), \
                            (w > w_cutoff_val)))

    date_spectrum_vent_dict = {}

    for var_name in spectrum_dict['data'].keys():
        meanfr = get_meanr_contribution_from_spectrum_var(var_name, adlr_dict, \
            spectrum_dict, cutoff_bins, incl_rain, incl_vent, HALO_bin_radii)
        meanfr = meanfr[filter_inds]
        date_spectrum_vent_dict[var_name] = meanfr

    return date_spectrum_vent_dict

def update_alldates_spectrum_vent_dict(alldates_spectrum_vent_dict, \
                                            date_spectrum_vent_dict):

    for var_name in date_spectrum_vent_dict.keys():
        alldates_spectrum_vent_dict[var_name] = np.concatenate(( \
            alldates_spectrum_vent_dict[var_name], \
            date_spectrum_vent_dict[var_name]))

    return alldates_spectrum_vent_dict

def make_vent_dsd_fig_with_wrf_for_date(date, \
    alldates_spectrum_vent_dict, case_label):

    fig, ax = plt.subplots()

    HALO_y_vals = np.array([np.nanmean(alldates_spectrum_vent_dict[key]) \
                            for key in alldates_spectrum_vent_dict.keys()])
    print(np.sum(HALO_y_vals))
    ax.plot(HALO_bin_radii*1.e6, \
            HALO_y_vals/HALO_log_bin_widths, \
            color=colors_dict['halo'], label='HALO 1500-2500 m')
    for versionstr in versionstrs:
        wrf_dsd_dict, wrf_vent_dsd_dict = get_wrf_dsd_dicts(case_label, versionstr)
        #print(wrf_vent_dsd_dict.keys())
        print(np.sum(wrf_vent_dsd_dict['data']*WRF_bin_radii))
        ax.plot(WRF_bin_radii*1.e6,
                wrf_vent_dsd_dict['data']*WRF_bin_radii/WRF_log_bin_widths, \
                color=colors_dict[case_color_key_dict[case_label]], \
                linestyle=linestyles_dict[versionstr], \
                label='WRF ' + legends_dict[versionstr])

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(r'$\frac{d(r \cdot N(r))}{d\log r}$ ($\frac{\mu m}{cm^3}$)')
    #ax.set_ylabel(r'$\frac{d(r^3 \cdot N(r))}{d\log r}$ ($\frac{\mu m^3}{cm^3}$)')

    ax.set_xscale('log')
    #ax.set_ylim([0, 20])
    #ax.set_yscale('log')

    ax.legend()
    ax.set_title('HALO vs ' + case_label + ' WRF simulation')
    
    outfile = FIG_DIR + 'no_vent_halo_slice_meanr_dsd_' + date + '_' + \
                        case_label + '_figure.png'
                            #case_label + '_figure.png'
                            #case_label + '_zoom2_figure.png'
                            #case_label + '_logy_figure.png'
                        #case_label + '_mass_logy_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_wrf_dsd_dicts(case_label, versionstr):

    print(case_label)

    case_dsd_filename = None#WRF_DATA_DIR + versionstr + 'dsd_dict_slice_' + case_label + '_data.npy'
    #case_vent_dsd_filename = WRF_DATA_DIR + versionstr + 'vent_dsd_dict_slice_' \
    #                            + case_label + '_data.npy'
    case_vent_dsd_filename = WRF_DATA_DIR + versionstr + 'niri_avg_' \
                                + case_label + '_data.npy'

    dsd_dict = None#np.load(case_dsd_filename, allow_pickle=True).item()
    vent_dsd_dict = np.load(case_vent_dsd_filename, allow_pickle=True).item()

    return dsd_dict, vent_dsd_dict

if __name__ == "__main__":
    main()
