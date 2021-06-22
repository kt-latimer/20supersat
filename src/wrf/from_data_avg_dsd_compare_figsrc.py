"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from halo import CAS_bins, CIP_bins
from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'halo': colors_arr[2], 'wrf_poll': colors_arr[5], \
                                    'wrf_unpoll': colors_arr[8]}
versionstrs = ['v1_', 'v2_']
                            
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
case_color_key_dict = {'Polluted': 'wrf_poll', \
                        'Unpolluted': 'wrf_unpoll'}

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2. 
CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)
CIP_bin_radii = np.sqrt((CIP_bins['upper']*CIP_bins['lower'])/4.)

def main():
    
    for case_label in case_label_dict.keys():
        fig, ax = plt.subplots()
        for versionstr in versionstrs:
            dsd_dict, vent_dsd_dict = get_dsd_dicts(case_label, versionstr)
            ax.plot(bin_radii*1.e6, vent_dsd_dict['mean']*bin_radii, \
                    color=colors_dict[case_color_key_dict[case_label]], \
                    label='WRF ' + case_label)
        ax.plot(halo_bin_radii*1.e6, halo_vent_dsd*halo_bin_radii, \
                            color=colors_dict['halo'], label='HALO')
        ax.set_xscale('log')

        outfile = FIG_DIR + 'avg_vent_dsd_slice_compare_' + \
                                    case_label + '_figure.png'
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()    
    
    #make_and_save_avg_dsd_comparison_plot(combined_vent_dsd_dict)

def get_dsd_dicts(case_label, versionstr):

    print(case_label)

    case_dsd_filename = DATA_DIR + versionstr + 'dsd_dict_slice_' + case_label + '_data.npy'
    case_vent_dsd_filename = DATA_DIR + versionstr + 'vent_dsd_dict_slice_' \
                                + case_label + '_data.npy'

    dsd_dict = np.load(case_dsd_filename, allow_pickle=True).item()
    vent_dsd_dict = np.load(case_vent_dsd_filename, allow_pickle=True).item()

    return dsd_dict, vent_dsd_dict

def make_and_save_avg_dsd_comparison_plot(combined_vent_dsd_dict):

    fig, ax = plt.subplots()

    ax.plot(bin_radii*1.e6, combined_vent_dsd_dict['Polluted']['mean']*bin_radii)
    #ax.fill_between(bin_radii*1.e6, \
    #    combined_vent_dsd_dict['Polluted']['mean']*bin_radii - \
    #    combined_vent_dsd_dict['Polluted']['std']*bin_radii, \
    #    combined_vent_dsd_dict['Polluted']['mean']*bin_radii + \
    #    combined_vent_dsd_dict['Polluted']['std']*bin_radii, \
    #    color='b', alpha=0.5)
    ax.plot(bin_radii*1.e6, combined_vent_dsd_dict['Unpolluted']['mean']*bin_radii)
    #ax.fill_between(bin_radii*1.e6, \
    #    combined_vent_dsd_dict['Unpolluted']['mean']*bin_radii - \
    #    combined_vent_dsd_dict['Unpolluted']['std']*bin_radii, \
    #    combined_vent_dsd_dict['Unpolluted']['mean']*bin_radii + \
    #    combined_vent_dsd_dict['Unpolluted']['std']*bin_radii, \
    #    color='orange', alpha=0.5)

    ax.set_xscale('log')
    #ax.set_yscale('log')

    outfile = FIG_DIR + versionstr + 'avg_vent_dsd_slice_compare_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

if __name__ == "__main__":
    main()
