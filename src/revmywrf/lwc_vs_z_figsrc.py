"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors

lwc_filter_val = 1.e-4
w_cutoff = 2

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

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

    for case_label in case_label_dict.keys():
        avg_lwc, avg_z, z_bins = get_case_data(case_label)
        make_and_save_lwc_vs_z_plot(avg_lwc, avg_z, z_bins, \
                                    case_label, versionstr)
    
def get_case_data(case_label):

    case_dir_name = case_label_dict[case_label]

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
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    z_bins = get_z_bins(z)

    filter_inds = np.logical_and.reduce((
                    #(lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    lwc = lwc[filter_inds]
    z = z[filter_inds]

    avg_lwc, avg_z = get_avg_lwc_and_z(lwc, z, z_bins)

    del lwc, temp, w, z #for memory

    return avg_lwc, avg_z, z_bins

def make_and_save_lwc_vs_z_plot(avg_lwc, avg_z, z_bins, case_label, versionstr):

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 15)

    ax.plot(avg_lwc, avg_z, color=colors_arr[5], linewidth=6)
    ax.set_xlabel(r'Avg LWC (kg/kg)')
    ax.set_ylabel(r'z (m)')

    outfile = FIG_DIR + versionstr + 'lwc_vs_z_' + case_label + '_figure.png'
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

def get_avg_lwc_and_z(lwc, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_lwc = np.zeros(n_bins)
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
            avg_lwc[i] = np.nan
            avg_z[i] = np.nan
        else:
            lwc_slice = lwc[bin_filter]
            z_slice = z[bin_filter]
            avg_lwc[i] = np.nanmean(lwc_slice)
            avg_z[i] = np.nanmean(z_slice)

    return avg_lwc, avg_z

if __name__ == "__main__":
    main()
