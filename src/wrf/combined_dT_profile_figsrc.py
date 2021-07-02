"""
Plot buoyancy profiles relative to hypothetical adiabatic SS=0 parcel for both
simulation and experimental datasets
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

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

#hard-coded based on shared_z_lim computed in ss_pred_vs_z
z_min = 973.0825
z_max = 4529.355
z_lim = (z_min, z_max)

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
g = 9.8 #grav accel (m/s^2)
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))

def main():

    ### halo ###
    halo_dict = np.load(HALO_DATA_DIR + 'dT_profile_data_alldates.npy', \
                        allow_pickle=True).item()
    avg_dT_halo = halo_dict['dT']
    avg_temp_halo = halo_dict['temp']
    avg_z_halo = halo_dict['z']
    z_bins_halo = halo_dict['z_bins']
    #in_z_lim_inds = get_in_z_lim_inds(avg_z_halo, z_lim)
    #avg_dT_halo = avg_dT_halo[in_z_lim_inds] 
    #avg_temp_halo = avg_temp_halo[in_z_lim_inds]
    #avg_z_halo = avg_z_halo[in_z_lim_inds]
    #in_z_lim_inds_bins = get_in_z_lim_inds_bins(in_z_lim_inds)
    #z_bins_halo = z_bins_halo[in_z_lim_inds_bins]

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
        #in_z_lim_inds = get_in_z_lim_inds(avg_z, z_lim)
        #avg_dT = avg_dT[in_z_lim_inds] 
        #avg_temp = avg_temp[in_z_lim_inds]
        #avg_z = avg_z[in_z_lim_inds]
        #in_z_lim_inds_bins = get_in_z_lim_inds_bins(in_z_lim_inds)
        #z_bins = z_bins[in_z_lim_inds_bins]

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

    new_z_bins = []
    found_first_edge = False

    print(np.shape(z_bins[:-1]))
    for i, lower_bin_edge in enumerate(z_bins[:-1]):
        upper_bin_edge = z_bins[i+1]
        if not found_first_edge:
            if upper_bin_edge < z_min:
                continue
            else:
                new_z_bins.append(z_min)
                start_ind = i
                found_first_edge = True
        else:
            print(i)
            new_z_bins.append(lower_bin_edge)
            if upper_bin_edge > z_max:
                new_z_bins.append(z_max)
                end_ind = i + 1
                break
            elif i == np.shape(z_bins[:-1])[0] - 1:
                new_z_bins.append(upper_bin_edge)
                end_ind = i + 1

    new_z_bins = np.array(new_z_bins)
    dz = np.array([new_z_bins[i+1] - new_z_bins[i] for i in \
                            range(np.shape(new_z_bins)[0] - 1)])

    print(np.shape(dz))
    print(np.shape(avg_dT))
    print(np.shape(avg_temp))
    print(start_ind, end_ind)
    dCAPE = np.nansum(g*dz*avg_dT[start_ind:end_ind]/avg_temp[start_ind:end_ind])

    print(label)
    print(z_bins[0], z_bins[-1])
    print(dz)
    print(avg_dT)
    print(avg_temp)
    print(dCAPE)

def make_and_save_dT_prof(dT_wrf_unpolluted, z_wrf_unpolluted, \
                            dT_wrf_polluted, z_wrf_polluted, \
                            dT_halo, z_halo):

    fig, ax = plt.subplots()

    ax.plot(dT_wrf_polluted, z_wrf_polluted, color=colors_dict['wrf_poll'], \
            label='WRF Polluted')
    ax.plot(dT_wrf_unpolluted, z_wrf_unpolluted, color=colors_dict['wrf_unpoll'], \
            label='WRF Unpolluted')
    #ax.plot(dT_halo, z_halo, color=colors_dict['halo'], \
    #        linestyle='-', marker='o', label='HALO')
    ax.plot(dT_halo, z_halo, color=colors_dict['halo'], \
            linestyle='-', label='HALO')
    #ax.plot(dT_caipeex, z_caipeex, color=colors_dict['caipeex'], \
    #        linestyle='-', marker='o', label='CAIPEEX')

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

def get_in_z_lim_inds(z_vals, z_lim):

    return np.logical_and(z_vals > z_lim[0], z_vals < z_lim[1])

def get_in_z_lim_inds_bins(in_z_lim_inds):

    in_z_lim_inds_bins = in_z_lim_inds.copy()
    for i, val in enumerate(in_z_lim_inds):
        if not val and in_z_lim_inds[i-1]:
           in_z_lim_inds_bins = np.insert(in_z_lim_inds_bins, i, True)

    return in_z_lim_inds_bins

if __name__ == "__main__":
    main()
