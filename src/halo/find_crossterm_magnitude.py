"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_ss_vs_t, get_lwc_vs_t, \
                            get_full_spectrum_dict, \
                            get_ss_qss_components

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[2]
magma_pink = colors_arr[5]
magma_orange = colors_arr[8]

splice_methods = ['cas_over_cip', 'cip_over_cas', 'wt_avg']

lwc_filter_val = 1.e-4 #10**(-3.5)
w_cutoff = 1

rmax = 102.e-6

z_min = -100
z_max = 6500

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

##
## physical constants
##
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_qss_alldates = np.array([])
    A_alldates = np.array([])
    B_alldates = np.array([])
    meanr_alldates = np.array([])
    nconc_alldates = np.array([])
    w_alldates = np.array([])

    for date in good_dates:
        ss_qss, A, B, meanr, nconc, w = get_ss_qss_data(date)
        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)
        A_alldates = add_to_alldates_array(A, A_alldates)
        B_alldates = add_to_alldates_array(B, B_alldates)
        meanr_alldates = add_to_alldates_array(meanr, meanr_alldates)
        nconc_alldates = add_to_alldates_array(nconc, nconc_alldates)
        w_alldates = add_to_alldates_array(w, w_alldates)

    print(np.shape(ss_qss_alldates))
    print(np.nanmean(ss_qss_alldates))
    print(np.mean(A_alldates*w_alldates/(4*np.pi*B_alldates))*np.mean(1./(meanr_alldates*nconc_alldates))*100.)
    print(np.mean(A_alldates*w_alldates/(4*np.pi*B_alldates))/np.mean((meanr_alldates*nconc_alldates))*100.)
    print(np.mean(A_alldates*w_alldates/(4*np.pi*B_alldates)))
    print(np.mean((meanr_alldates*nconc_alldates)))

def add_to_alldates_array(arr, arr_alldates):

    if arr_alldates is None:
        return arr
    else:
        return np.concatenate((arr_alldates, arr))

def get_ss_qss_data(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(cas_dict, \
                                cip_dict, change_cas_corr)

    lwc = get_lwc_vs_t(adlr_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']
    z = adlr_dict['data']['alt']
    #print('one')
    ss_qss = get_ss_vs_t(adlr_dict, full_spectrum_dict, change_cas_corr, \
                    cutoff_bins, full_ss, incl_rain, incl_vent)
    #print('two')
    A, B, meanr, nconc = get_ss_qss_components(adlr_dict, full_spectrum_dict, \
            change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (z > 1500), \
                    (z < 2500), \
                    (temp > 273)))

    if np.sum(filter_inds) != 0:
        ss_qss = ss_qss[filter_inds]
        A = A[filter_inds]
        B = B[filter_inds]
        meanr = meanr[filter_inds]
        nconc = nconc[filter_inds]
        w = w[filter_inds]
    else:
        ss_qss = np.array([])
        A = np.array([])
        B = np.array([])
        meanr = np.array([])
        nconc = np.array([])
        w = np.array([])

    return ss_qss, A, B, meanr, nconc, w

if __name__ == "__main__":
    main()
