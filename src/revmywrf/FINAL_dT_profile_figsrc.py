"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

lwc_filter_val = 1.e-4
w_cutoff = 2

z_min = -100
z_max = 6500

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
g = 9.8 #grav accel (m/s^2)
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))

cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    pres_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    temp_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    ss_qss_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    w_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_dict = {'allpts': {'Polluted': None, 'Unpolluted': None}, \
                   'up10perc': {'Polluted': None, 'Unpolluted': None}}
    z_bins_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        pres, temp, ss_qss, w, z, z_bins = get_case_data(case_label)
        pres_dict['allpts'][case_label] = pres
        temp_dict['allpts'][case_label] = temp
        ss_qss_dict['allpts'][case_label] = ss_qss
        w_dict['allpts'][case_label] = w
        z_dict['allpts'][case_label] = z
        z_bins_dict[case_label] = z_bins

        pres, temp, ss_qss, w, z = \
            get_up10perc_data(pres, temp, ss_qss, w, z)
        pres_dict['up10perc'][case_label] = pres
        temp_dict['up10perc'][case_label] = temp
        ss_qss_dict['up10perc'][case_label] = ss_qss
        w_dict['up10perc'][case_label] = w
        z_dict['up10perc'][case_label] = z

    make_and_save_dT_profile(pres_dict['allpts'], \
        temp_dict['allpts'], ss_qss_dict['allpts'], \
        w_dict['allpts'], z_dict['allpts'], z_bins_dict, 'allpts')
    make_and_save_dT_profile(pres_dict['up10perc'], \
        temp_dict['up10perc'], ss_qss_dict['up10perc'], \
        w_dict['up10perc'], z_dict['up10perc'], z_bins_dict, 'up10perc')

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
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #before filtering, get z bins
    z_bins = get_z_bins(z)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    temp = temp[filter_inds]
    w = w[filter_inds]
    z = z[filter_inds]

    return pres, temp, ss_qss, w, z, z_bins

def get_up10perc_data(pres, temp, ss_qss, w, z):

    w_cutoff = np.percentile(w, 90)
    up10perc_inds = w > w_cutoff

    pres = pres[up10perc_inds]
    temp = temp[up10perc_inds]
    ss_qss = ss_qss[up10perc_inds]
    w = w[up10perc_inds]
    z = z[up10perc_inds]

    return pres, temp, ss_qss, w, z

def make_and_save_dT_profile(pres_dict, temp_dict, ss_qss_dict, \
                            w_dict, z_dict, z_bins_dict, label):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 12)
    linestyle_str_dict = {'Polluted': '-', 'Unpolluted': '--'}

    for case_label in case_label_dict.keys():
        pres = pres_dict[case_label]
        temp = temp_dict[case_label]
        ss_qss = ss_qss_dict[case_label]
        w = w_dict[case_label]
        z = z_dict[case_label]
        z_bins = z_bins_dict[case_label]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])
        qvstar = get_qvstar(pres, temp)
        color = colors_dict[label]

        dT = get_dT(qvstar, ss_qss, temp)
        avg_dT, avg_temp, avg_z = get_avg_dT_and_temp_and_z(dT, temp, z, z_bins)

        dCAPE = np.nansum(dz*g*avg_dT/avg_temp)
        print(case_label, label)
        print(dCAPE)

        linestyle_str = linestyle_str_dict[case_label]
        ax.plot(avg_dT, avg_z, linestyle=linestyle_str, \
                color=color, linewidth=6)

    #formatting
    ax.set_ylim((z_min, z_max))
    ax.yaxis.grid()
    ax.set_xlabel(r'dT (K)')
    ax.set_ylabel(r'z (m)')
    #formatter = ticker.ScalarFormatter(useMathText=True)
    #formatter.set_scientific(True) 
    #formatter.set_powerlimits((-1,1)) 
    #ax.xaxis.set_major_formatter(formatter)

    #custom legend
    poll_line = Line2D([0], [0], color=color, \
                        linewidth=6, linestyle='-')
    unpoll_line = Line2D([0], [0], color=color, \
                        linewidth=6, linestyle='--')
    ax.legend([poll_line, unpoll_line], ['Polluted', 'Unpolluted'])

    outfile = FIG_DIR + versionstr + 'FINAL_dT_profile_' + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

def get_up10perc_inds(w):

    w_cutoff = np.percentile(w, 90)
    up10perc_inds = w > w_cutoff

    return up10perc_inds 

def get_dT(qvstar, ss_qss, temp):

    dRH = ss_qss/100. #assuming parcel has RH=1 (as a fraction not percent)
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

def get_z_bins(z):

    n_bins = np.shape(z)[1]
    n_edges = n_bins + 1
    avg_z = np.array([np.mean(z[:, i, :, :]) for i in range(n_bins)])
    z_bins = [] 

    for i in range(1, n_bins):
        layer_geom_mean = np.sqrt(avg_z[i-1]*avg_z[i])
        if layer_geom_mean < z_max:
            z_bins.append(layer_geom_mean)
        else:
            break

    z_bins.insert(0, avg_z[0]*np.sqrt(avg_z[0]/avg_z[1]))

    return np.array(z_bins)

if __name__ == "__main__":
    main()
