"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_ss_vs_t, get_lwc

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
incl_rain = False 
incl_vent = False
full_ss = True

def main():
    
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    pres_alldates = None 
    ss_qss_alldates = None
    temp_alldates = None
    w_alldates = None
    z_alldates = None

    for date in good_dates:
        pres, temp, ss_qss, w, z = get_one_day_data(date)
        pres_alldates = add_to_alldates_array(pres, pres_alldates)
        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)
        temp_alldates = add_to_alldates_array(temp, temp_alldates)
        w_alldates = add_to_alldates_array(w, w_alldates)
        z_alldates = add_to_alldates_array(z, z_alldates)

    h_z, z_bins = np.histogram(z_alldates, bins=30, density=True)
    print(z_bins)

    make_and_save_dT_profile(pres_alldates, temp_alldates, \
            ss_qss_alldates, w_alldates, z_alldates, z_bins)

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def get_one_day_data(date):

    metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
    met_dict = np.load(metfile, allow_pickle=True).item()
    cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cpd_dict = np.load(cpdfile, allow_pickle=True).item()

    lwc = get_lwc(cpd_dict,cutoff_bins)
    pres = met_dict['data']['pres']
    temp = met_dict['data']['temp']
    w = met_dict['data']['w']
    z = met_dict['data']['alt']
    ss_qss = get_ss_vs_t(met_dict, cpd_dict, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #there's a weird outlier which the third line removes
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (ss_qss < 100), \
                    (temp > 273)))

    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    temp = temp[filter_inds]
    w = w[filter_inds]
    z = z[filter_inds]

    return pres, temp, ss_qss, w, z

def make_and_save_dT_profile(pres, temp, ss_qss, w, z, z_bins):

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 12)

    dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                    range(np.shape(z_bins)[0] - 1)])
    qvstar = get_qvstar(pres, temp)

    inds_dict = {'allpts': np.array([True for i in range(np.shape(z)[0])]), \
                    'up10perc': get_up10perc_inds(w)}

    for key in inds_dict.keys():
        color = colors_dict[key]
        inds = inds_dict[key]

        dT = get_dT(qvstar[inds], ss_qss[inds], temp[inds])
        avg_dT, avg_temp, avg_z = get_avg_dT_and_temp_and_z(dT, temp[inds], z[inds], z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_dT))
        avg_dT = avg_dT[notnan_inds]
        avg_temp = avg_temp[notnan_inds]
        avg_z = avg_z[notnan_inds]

        dCAPE = np.nansum(dz[notnan_inds]*g*avg_dT/avg_temp)
        print(key)
        print(dCAPE)

        ax.plot(avg_dT, avg_z, linestyle='-', marker='o', \
                color=color, linewidth=6, markersize=17)

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
    allpts_line = Line2D([0], [0], color=colors_dict['allpts'], \
                        linewidth=6, linestyle='-')
    up10perc_line = Line2D([0], [0], color=colors_dict['up10perc'], \
                        linewidth=6, linestyle='-')
    ax.legend([allpts_line, up10perc_line], ['All cloudy updrafts', \
                                    'Top 10% cloudy updrafts (by w)'])

    outfile = FIG_DIR + versionstr + 'FINAL_dT_profile_figure.png'
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

if __name__ == "__main__":
    main()
