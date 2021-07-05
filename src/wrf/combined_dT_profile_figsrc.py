"""
Plot buoyancy profiles relative to hypothetical adiabatic SS=0 parcel for both
simulation and experimental datasets
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from phys_consts import *
from wrf import DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_ss_pred, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('viridis', 10).colors
colors_dict = {'halo': colors_arr[2], 'wrf_poll': colors_arr[5], \
                                    'wrf_unpoll': colors_arr[8]}

HALO_DATA_DIR = '/global/home/users/kalatimer/proj/20supersat/data/halo/'
CAIPEEX_DATA_DIR = \
    '/global/home/users/kalatimer/proj/20supersat/data/caipeex/'

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

#hard-coded based on get_z_lims.py files for both wrf and halo 
z_min = 731.6867 
z_max = 4490.85 
z_lim = (z_min, z_max)

def main():

    ### halo ###
    halo_dict = np.load(HALO_DATA_DIR + 'dT_profile_data.npy', \
                        allow_pickle=True).item()
    avg_dT_halo = halo_dict['dT']
    avg_temp_halo = halo_dict['temp']
    avg_z_halo = halo_dict['z']
    z_bins_halo = halo_dict['z_bins']

    ### wrf ###
    wrf_dict = np.load(DATA_DIR + 'filtered_data_dict.npy', \
                        allow_pickle=True).item()

    dT_and_z_dict = {'Polluted': None, 'Unpolluted': None}
    
    for case_label in case_label_dict.keys():
        case_filtered_data_dict = wrf_dict[case_label]
        pres = case_filtered_data_dict['pres']
        ss_qss = case_filtered_data_dict['ss_qss']
        temp = case_filtered_data_dict['temp']
        z = case_filtered_data_dict['z']
        z_bins = case_filtered_data_dict['z_bins']

        qvstar = get_qvstar(pres, temp)
        ss_pred = get_ss_pred(ss_qss)

        dT = get_dT(qvstar, ss_pred, temp)
        avg_dT, avg_temp, avg_z = \
                get_avg_dT_and_temp_and_z(dT, temp, z, z_bins)

        dT_and_z_dict[case_label] = {'dT': avg_dT, 'z': avg_z}
    
    calc_and_print_dCAPE(avg_dT_halo, avg_temp_halo, z_bins_halo, 'halo')
    calc_and_print_dCAPE(dT_and_z_dict['Unpolluted']['dT'], \
                            wrf_dict['Unpolluted']['temp'], \
                            wrf_dict['Unpolluted']['z_bins'], 'unpolluted')
    calc_and_print_dCAPE(dT_and_z_dict['Polluted']['dT'], \
                            wrf_dict['Polluted']['temp'], \
                            wrf_dict['Polluted']['z_bins'], 'polluted')

    make_and_save_dT_prof(dT_and_z_dict['Unpolluted']['dT'], \
                            dT_and_z_dict['Unpolluted']['z'], \
                            dT_and_z_dict['Polluted']['dT'], \
                            dT_and_z_dict['Polluted']['z'], \
                            avg_dT_halo, avg_z_halo)

def calc_and_print_dCAPE(avg_dT, avg_temp, z_bins, label):

    dz = get_dz_from_shared_z_lim(z_bins)
    dCAPE = np.nansum(g*dz*avg_dT/avg_temp)

    print(label)
    print(dCAPE)

def make_and_save_dT_prof(dT_wrf_unpolluted, z_wrf_unpolluted, \
                            dT_wrf_polluted, z_wrf_polluted, \
                            dT_halo, z_halo):

    fig, ax = plt.subplots()

    ax.plot(dT_wrf_polluted, z_wrf_polluted, color=colors_dict['wrf_poll'], \
            label='WRF Polluted')
    ax.plot(dT_wrf_unpolluted, z_wrf_unpolluted, color=colors_dict['wrf_unpoll'], \
            label='WRF Unpolluted')
    ax.plot(dT_halo, z_halo, color=colors_dict['halo'], \
            linestyle='-', label='HALO')

    ax.set_ylim(z_lim)
    ax.set_xlabel(r'$\Delta T$ (K)')
    ax.set_ylabel(r'z (m)')
    plt.legend()
    fig.suptitle('Maximum warm-phase invigoration')

    outfile = FIG_DIR + 'combined_dT_profile_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

def get_dT(qvstar, ss_pred, temp):

    dRH = ss_pred/100. #assuming parcel has RH=1 (as a fraction not percent)
    dT = qvstar*L_v/(C_ap + qvstar*L_v**2./(R_v*temp**2.))*dRH

    return dT

def get_avg_dT_and_temp_and_z(dT, temp, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_dT = np.zeros(n_bins)
    avg_temp = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        if i == n_bins-1: #last upper bin edge is inclusive
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z <= upper_bin_edge)))
        else: 
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_dT[i] = np.nan
            avg_temp[i] = np.nan
            avg_z[i] = np.nan
        else:
            dT_slice = dT[bin_filter]
            temp_slice = temp[bin_filter]
            z_slice = z[bin_filter]
            avg_dT[i] = np.nanmean(dT_slice)
            avg_temp[i] = np.nanmean(temp_slice)
            avg_z[i] = np.nanmean(z_slice)

    return avg_dT, avg_temp, avg_z

def get_dz_from_shared_z_lim(z_bins):

    dz = [z_bins[i+1] - z_bins[i] for i in range(len(z_bins[:-1]))]

    for i, lo_bin_edge in enumerate(z_bins[:-1]):
        up_bin_edge = z_bins[i+1]
        if up_bin_edge < z_lim[0] or lo_bin_edge > z_lim[1]:
            dz[i] = 0
        elif z_lim[0] > up_bin_edge and z_lim[0] < up_bin_edge:
            dz[i] = up_bin_edge - z_lim[0]
        elif z_lim[1] > up_bin_edge and z_lim[1] < up_bin_edge:
            dz[i] = z_lim[1] - lo_bin_edge

    return dz

if __name__ == "__main__":
    main()
