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
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = False 

CAS_bin_radii = (CAS_bins['upper'] + CAS_bins['lower'])/4.
CIP_bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        make_dsd_figs_for_date(date)

def make_dsd_figs_for_date(date):

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

    cas_dsd_dict, cip_dsd_dict = get_dsd_dicts(adlr_dict, cas_dict, \
                                                cip_dict, filter_inds)
    make_dsd_fig(cas_dsd_dict, cip_dsd_dict, date)

def get_dsd_dicts(adlr_dict, cas_dict, cip_dict, filter_inds):

    cas_dsd_dict = {'nconc': {'mean': [], 'std': [], 'median': [], \
                    'up_quart': [], 'lo_quart': []}, \
                    'meanfr': {'mean': [], 'std': [], 'median': [], \
                    'up_quart': [], 'lo_quart': []}}
    cip_dsd_dict = {'nconc': {'mean': [], 'std': [], 'median': [], \
                    'up_quart': [], 'lo_quart': []}, \
                    'meanfr': {'mean': [], 'std': [], 'median': [], \
                    'up_quart': [], 'lo_quart': []}}

    for j in range(5, 17):
        var_name = 'nconc_' + str(j)
        if change_cas_corr:
            var_name += '_corr'
        nconc = get_nconc_contribution_from_cas_var(var_name, adlr_dict, \
                cas_dict, change_cas_corr, cutoff_bins, incl_rain, incl_vent)
        cas_dsd_dict['nconc']['mean'].append( \
                np.mean(nconc[filter_inds]))
        cas_dsd_dict['nconc']['std'].append( \
                np.std(nconc[filter_inds]))
        cas_dsd_dict['nconc']['median'].append( \
                np.median(nconc[filter_inds]))
        cas_dsd_dict['nconc']['up_quart'].append( \
                np.percentile(nconc[filter_inds], 75) \
                - cas_dsd_dict['nconc']['median'][j - 5])
        cas_dsd_dict['nconc']['lo_quart'].append( \
                (np.percentile(nconc[filter_inds], 25) \
                - cas_dsd_dict['nconc']['median'][j - 5])*-1)
        meanfr = get_meanr_contribution_from_cas_var(var_name, adlr_dict, \
                cas_dict, change_cas_corr, cutoff_bins, incl_rain, incl_vent)
        cas_dsd_dict['meanfr']['mean'].append( \
                np.mean(meanfr[filter_inds]))
        cas_dsd_dict['meanfr']['std'].append( \
                np.std(meanfr[filter_inds]))
        cas_dsd_dict['meanfr']['median'].append( \
                np.median(meanfr[filter_inds]))
        cas_dsd_dict['meanfr']['up_quart'].append( \
                np.percentile(meanfr[filter_inds], 75) \
                - cas_dsd_dict['meanfr']['median'][j - 5])
        cas_dsd_dict['meanfr']['lo_quart'].append( \
                (np.percentile(meanfr[filter_inds], 25) \
                - cas_dsd_dict['meanfr']['median'][j - 5])*-1)

    for j in range(1, 20):
        var_name = 'nconc_' + str(j)
        nconc = get_nconc_contribution_from_cip_var(var_name, adlr_dict, \
                cip_dict)
        cip_dsd_dict['nconc']['mean'].append( \
                np.mean(nconc[filter_inds]))
        cip_dsd_dict['nconc']['std'].append( \
                np.std(nconc[filter_inds]))
        cip_dsd_dict['nconc']['median'].append( \
                np.median(nconc[filter_inds]))
        cip_dsd_dict['nconc']['up_quart'].append( \
                np.percentile(nconc[filter_inds], 75) \
                - cip_dsd_dict['nconc']['median'][j - 1])
        cip_dsd_dict['nconc']['lo_quart'].append( \
                (np.percentile(nconc[filter_inds], 25) \
                - cip_dsd_dict['nconc']['median'][j - 1])*-1)
        meanfr = get_meanr_contribution_from_cip_var(var_name, adlr_dict, \
                cip_dict, incl_rain, incl_vent)
        cip_dsd_dict['meanfr']['mean'].append( \
                np.mean(meanfr[filter_inds]))
        cip_dsd_dict['meanfr']['std'].append( \
                np.std(meanfr[filter_inds]))
        cip_dsd_dict['meanfr']['median'].append( \
                np.median(meanfr[filter_inds]))
        cip_dsd_dict['meanfr']['up_quart'].append( \
                np.percentile(meanfr[filter_inds], 75) \
                - cip_dsd_dict['meanfr']['median'][j - 1])
        cip_dsd_dict['meanfr']['lo_quart'].append( \
                (np.percentile(meanfr[filter_inds], 25) \
                - cip_dsd_dict['meanfr']['median'][j - 1])*-1)

    for outer_key in cas_dsd_dict.keys():
        for inner_key in cas_dsd_dict[outer_key].keys():
            cas_dsd_dict[outer_key][inner_key] = \
            np.array(cas_dsd_dict[outer_key][inner_key])

    for outer_key in cip_dsd_dict.keys():
        for inner_key in cip_dsd_dict[outer_key].keys():
            cip_dsd_dict[outer_key][inner_key] = \
            np.array(cip_dsd_dict[outer_key][inner_key])

    return cas_dsd_dict, cip_dsd_dict
    
def make_dsd_fig(cas_dsd_dict, cip_dsd_dict, date):

    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)

    ax11.errorbar(CAS_bin_radii*1.e6, cas_dsd_dict['nconc']['mean']*1.e-6, \
                            yerr=cas_dsd_dict['nconc']['std']*1.e-6, fmt='o')
    ax12.errorbar(CAS_bin_radii*1.e6, cas_dsd_dict['meanfr']['mean'], \
                            yerr=cas_dsd_dict['meanfr']['std'], fmt='o')
    ax21.errorbar(CAS_bin_radii*1.e6, cas_dsd_dict['nconc']['median']*1.e-6, \
                                yerr=[cas_dsd_dict['nconc']['lo_quart']*1.e-6, \
                                cas_dsd_dict['nconc']['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(CAS_bin_radii*1.e6, cas_dsd_dict['meanfr']['median'], \
                            yerr=[cas_dsd_dict['meanfr']['lo_quart'], \
                            cas_dsd_dict['meanfr']['up_quart']], fmt='o')

    ax11.errorbar(CIP_bin_radii*1.e6, cip_dsd_dict['nconc']['mean']*1.e-6, \
                            yerr=cip_dsd_dict['nconc']['std']*1.e-6, fmt='o')
    ax12.errorbar(CIP_bin_radii*1.e6, cip_dsd_dict['meanfr']['mean'], \
                            yerr=cip_dsd_dict['meanfr']['std'], fmt='o')
    ax21.errorbar(CIP_bin_radii*1.e6, cip_dsd_dict['nconc']['median']*1.e-6, \
                                yerr=[cip_dsd_dict['nconc']['lo_quart']*1.e-6, \
                                cip_dsd_dict['nconc']['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(CIP_bin_radii*1.e6, cip_dsd_dict['meanfr']['median'], \
                            yerr=[cip_dsd_dict['meanfr']['lo_quart'], \
                            cip_dsd_dict['meanfr']['up_quart']], fmt='o')

    ax11.set_xscale('log')
    ax12.set_xscale('log')
    ax21.set_xscale('log')
    ax22.set_xscale('log')
    ax11.set_yscale('log')
    ax12.set_yscale('log')
    ax21.set_yscale('log')
    ax22.set_yscale('log')
        
    outfile = FIG_DIR + 'avg_dsds_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
