"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo import CAS_bins, CIP_bins
from halo import DATA_DIR as HALO_DATA_DIR
from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.dsd_data_functions import get_bin_nconc
from wrf.ss_functions import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict = {'CAS': colors_arr[2], 'CIP': colors_arr[5], \
                                        'WRF': colors_arr[8]}
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

WRF_bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
WRF_bin_radii = WRF_bin_diams/2.

bin_bar_dict = {'CAS': {'bottom': [4, 4], 'top': [5, 5]}, \
                'CIP': {'bottom': [2, 2], 'top': [3, 3]}, \
                'WRF': [1 for r in WRF_bin_radii]}

def main():

    for case_label in case_label_dict.keys():
        cas_nconc_dict, cip_nconc_dict = get_halo_data()
        wrf_filter, wrf_dsd_vars, wrf_rho_air = get_wrf_data(case_label)

        for i, val in enumerate(CAS_bins['lower']):
            make_nconc_hist_for_bin(cas_nconc_dict, 'CAS', wrf_dsd_vars, \
                            wrf_filter, wrf_rho_air, i, case_label)

        for i, val in enumerate(CIP_bins['lower']):
            make_nconc_hist_for_bin(cip_nconc_dict, 'CIP', wrf_dsd_vars, \
                            wrf_filter, wrf_rho_air, i, case_label)

def get_halo_data():
    
    casfile = HALO_DATA_DIR + 'alldates_cas_nconc.npy'
    cipfile = HALO_DATA_DIR + 'alldates_cip_nconc.npy'

    cas_nconc_dict = np.load(casfile, allow_pickle=True).item()
    cip_nconc_dict = np.load(cipfile, allow_pickle=True).item()

    return cas_nconc_dict, cip_nconc_dict
    
def get_wrf_data(case_label):

    #get met input file variables (naming a bit confusing here but trying
    #to maintain consistency with make_net_vars code...)
    case_dir_name = case_label_dict[case_label]

    met_input_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_input_vars = met_input_file.variables

    #get relevant environmental data 
    rho_air_data = met_input_vars['rho_air'][...]
    temp = met_input_vars['temp'][...]
    w = met_input_vars['w'][...]

    filter_inds = np.logical_and(w > 1, temp > 273)

    del temp, w #for memory

    #close met input file for memory
    met_input_file.close()
    
    #get raw input file vars (with dsd data)
    dsd_input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    dsd_input_vars = dsd_input_file.variables

    return filter_inds, dsd_input_vars, rho_air_data

def make_nconc_hist_for_bin(nconc_dict, instr_name, wrf_dsd_vars, \
                            wrf_filter, wrf_rho_air, i, case_label):

    if instr_name == 'CAS':
        bin_dict = CAS_bins
        var_name = 'nconc_' + str(i+5) + '_corr'
    else:
        bin_dict = CIP_bins
        var_name = 'nconc_' + str(i+1)

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [7, 1]})

    nconc_dict[var_name] = \
        nconc_dict[var_name][np.logical_not(np.isinf(nconc_dict[var_name]))]

    ax0.hist(nconc_dict[var_name], bins=30, density=True, label=instr_name, \
            facecolor=(0, 0, 0, 0.0), edgecolor=colors_dict[instr_name], \
            histtype='stepfilled', linewidth=3)

    lo_bin_edge = bin_dict['lower'][i]
    hi_bin_edge = bin_dict['upper'][i]

    wrf_bins = get_wrf_bins_in_range(lo_bin_edge, hi_bin_edge)
    wrf_nconc = np.zeros(np.shape(wrf_rho_air[wrf_filter]))

    for j in wrf_bins:
        bin_nconc = get_bin_nconc(j+1, wrf_dsd_vars, wrf_rho_air)
        wrf_nconc += bin_nconc[wrf_filter]

    ax0.hist(wrf_nconc, bins=30, density=True, label='WRF', \
            facecolor=(0, 0, 0, 0.0), edgecolor=colors_dict['WRF'], \
            histtype='stepfilled', linewidth=3)

    #ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.legend()
    ax1 = make_bin_bar(ax1, instr_name, lo_bin_edge, hi_bin_edge, wrf_bins) 

    outfile = FIG_DIR + 'nconc_hist_by_bin_' + case_label + \
        '_' + instr_name + '_' + str(lo_bin_edge) + '_v2_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def make_bin_bar(ax1, instr_name, lo_bin_edge, hi_bin_edge, wrf_bins):

    ax1.spines['right'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.tick_params(which='major', width=1.00)
    ax1.tick_params(which='major', length=5)
    ax1.tick_params(which='minor', width=0.75)
    ax1.tick_params(which='minor', length=2.5)
    ax1.patch.set_alpha(0.0)

    ax1.scatter(WRF_bin_radii*1.e6, bin_bar_dict['WRF'], \
                c=colors_dict['WRF'], s=6, alpha=0.5)

    for i, CAS_lo_bin_edge in enumerate(CAS_bins['lower']):
        CAS_up_bin_edge = CAS_bins['upper'][i]
        ax1.fill_between([CAS_lo_bin_edge*1.e6, CAS_up_bin_edge*1.e6], \
            bin_bar_dict['CAS']['bottom'], bin_bar_dict['CAS']['top'], \
            color=colors_dict['CAS'], alpha=0.5)

    for i, CIP_lo_bin_edge in enumerate(CIP_bins['lower']):
        CIP_up_bin_edge = CIP_bins['upper'][i]
        ax1.fill_between([CIP_lo_bin_edge*1.e6, CIP_up_bin_edge*1.e6], \
            bin_bar_dict['CIP']['bottom'], bin_bar_dict['CIP']['top'], \
            color=colors_dict['CIP'], alpha=0.5)

    for j in wrf_bins:
        ax1.scatter([WRF_bin_radii[j]*1.e6], [bin_bar_dict['WRF'][j]], \
                                    c=colors_dict['WRF'], s=6, alpha=1)

    ax1.fill_between([lo_bin_edge*1.e6, hi_bin_edge*1.e6], \
                        bin_bar_dict[instr_name]['bottom'], \
                        bin_bar_dict[instr_name]['top'], \
                        color=colors_dict[instr_name], alpha=1)

    ax1.set_xscale('log')
    ax1.set_ylabel('Bins')
    ax1.set_ylim([bin_bar_dict['WRF'][0] - 0.5, \
        bin_bar_dict['CAS']['top'][0] + 0.5])

    return ax1

def get_wrf_bins_in_range(lo_bin_edge, hi_bin_edge):

    wrf_bins = []

    for i, r in enumerate(WRF_bin_radii):
        if r >= lo_bin_edge and r < hi_bin_edge:
            wrf_bins.append(i)

    return wrf_bins

if __name__ == "__main__":
    main()
