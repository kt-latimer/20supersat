"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict = {'poll': colors_arr[1], \
               'unpoll': colors_arr[3], \
               'halo': colors_arr[5], \
               'caipeex': colors_arr[7]}

lwc_filter_val = 1.e-4
w_cutoff = 2

HALO_DATA_DIR = '/global/home/users/kalatimer/proj/20supersat/data/revhalo/'
CAIPEEX_DATA_DIR = \
    '/global/home/users/kalatimer/proj/20supersat/data/revcaipeex/'

cutoff_bins = True
incl_rain = True
incl_vent = True
full_ss = True

z_max_wrf = 6500

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
    halo_dict = np.load(HALO_DATA_DIR + 'v24_dT_profile_data_alldates.npy', \
                        allow_pickle=True).item()
    avg_dT_halo = halo_dict['dT']
    avg_temp_halo = halo_dict['temp']
    avg_z_halo = halo_dict['z']
    z_bins_halo = halo_dict['z_bins']

    ### caipeex ###
    caipeex_dict = np.load(CAIPEEX_DATA_DIR + 'v10_dT_profile_data_alldates.npy', \
                            allow_pickle=True).item()
    avg_dT_caipeex = caipeex_dict['dT']
    avg_temp_caipeex = caipeex_dict['temp']
    avg_z_caipeex = caipeex_dict['z']
    z_bins_caipeex = caipeex_dict['z_bins']

    calc_and_print_dCAPE(avg_dT_halo, avg_temp_halo, z_bins_halo, 'halo')
    calc_and_print_dCAPE(avg_dT_caipeex, avg_temp_caipeex, z_bins_caipeex, 'caipeex')
    return
    
    #jk THIS is the laziest code of my life!

    ### wrf unpolluted ###
    case_dir_name = 'C_PI/'

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    z_bins_unpolluted = get_z_bins(z)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    del lwc, w #for memory

    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    temp = temp[filter_inds]
    z = z[filter_inds]

    qvstar = get_qvstar(pres, temp)

    dT = get_dT(qvstar, ss_qss, temp)

    del pres, qvstar, ss_qss #for memory

    avg_dT_unpolluted, avg_temp_unpolluted, avg_z_unpolluted = \
                    get_avg_dT_and_temp_and_z(dT, temp, z, z_bins_unpolluted)

    del dT, temp, z #for memory

    ### wrf polluted ###
    case_dir_name = 'C_BG/'

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    z_bins_polluted = get_z_bins(z)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    del lwc, w #for memory

    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    temp = temp[filter_inds]
    z = z[filter_inds]

    qvstar = get_qvstar(pres, temp)

    dT = get_dT(qvstar, ss_qss, temp)

    del pres, qvstar, ss_qss #for memory

    avg_dT_polluted, avg_temp_polluted, avg_z_polluted = \
                    get_avg_dT_and_temp_and_z(dT, temp, z, z_bins_polluted)

    del dT, temp, z #for memory
    
    calc_and_print_dCAPE(avg_dT_halo, avg_temp_halo, z_bins_halo, 'halo')
    calc_and_print_dCAPE(avg_dT_caipeex, avg_temp_caipeex, z_bins_caipeex, 'caipeex')
    calc_and_print_dCAPE(avg_dT_unpolluted, avg_temp_unpolluted, z_bins_unpolluted, 'unpolluted')
    calc_and_print_dCAPE(avg_dT_polluted, avg_temp_polluted, z_bins_polluted, 'polluted')
    make_and_save_dT_prof(avg_dT_polluted, avg_dT_unpolluted, avg_z_polluted, \
                            avg_z_unpolluted, avg_dT_halo, avg_dT_caipeex, \
                            avg_z_halo, avg_z_caipeex, versionstr)

def calc_and_print_dCAPE(avg_dT, avg_temp, z_bins, label):

    print(np.shape(avg_dT))
    print(np.shape(avg_temp))
    print(np.shape(z_bins))
    print(z_bins)
    z_min = 647.
    z_max_shared = 4488.

    new_z_bins = []
    found_first_edge = False

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
            new_z_bins.append(lower_bin_edge)
            if upper_bin_edge > z_max_shared:
                new_z_bins.append(z_max_shared)
                end_ind = i + 1
                break

    print(start_ind, end_ind)
    print(new_z_bins)
    new_z_bins = np.array(new_z_bins)
    dz = np.array([new_z_bins[i+1] - new_z_bins[i] for i in \
                            range(np.shape(new_z_bins)[0] - 1)])
    print(dz)

    dCAPE = np.nansum(g*dz*avg_dT[start_ind:end_ind]/avg_temp[start_ind:end_ind])

    print(label)
    print(dCAPE)

def make_and_save_dT_prof(dT_polluted, dT_unpolluted, z_polluted, \
                            z_unpolluted, dT_halo, dT_caipeex, z_halo, \
                            z_caipeex, versionstr):

    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)

    ax.plot(dT_polluted, z_polluted, color=colors_dict['poll'], \
            linewidth=6, label='WRF - polluted')
    ax.plot(dT_unpolluted, z_unpolluted, color=colors_dict['unpoll'], \
            linewidth=6, label='WRF - unpolluted')
    ax.plot(dT_halo, z_halo, color=colors_dict['halo'], \
            linestyle='-', marker='o', markersize=17, \
            linewidth=6, label='HALO')
    ax.plot(dT_caipeex, z_caipeex, color=colors_dict['caipeex'], \
            linestyle='-', marker='o', markersize=17, \
            linewidth=6, label='CAIPEEX')

    ax.set_xlabel(r'$\delta T$ (K)')
    ax.set_ylabel(r'z (m)')
    plt.legend()

    outfile = FIG_DIR + versionstr + 'FINAL_combined_dT_profile_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_z_bins(z):

    n_bins = np.shape(z)[1]
    n_edges = n_bins + 1
    avg_z = np.array([np.mean(z[:, i, :, :]) for i in range(n_bins)])
    z_bins = [] 

    for i in range(1, n_bins):
        layer_geom_mean = np.sqrt(avg_z[i-1]*avg_z[i])
        if layer_geom_mean < z_max_wrf:
            z_bins.append(layer_geom_mean)
        else:
            break

    z_bins.insert(0, avg_z[0]*np.sqrt(avg_z[0]/avg_z[1]))

    return np.array(z_bins)

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

def get_dT(qvstar, ss_qss, temp):

    dRH = ss_qss/100. #assuming parcel has RH=1 (as a fraction not percent)
    dT = qvstar*L_v/(C_ap + qvstar*L_v**2./(R_v*temp**2.))*dRH

    return dT

def get_avg_dT_and_temp_and_z(dT, temp, z, z_bins):

    print(np.shape(dT))
    print(np.shape(temp))
    print(np.shape(z))
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
        print(np.shape(bin_filter))
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

if __name__ == "__main__":
    main()
