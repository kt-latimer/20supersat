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
from halo.utils import linregress, match_multiple_arrays

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        print(date)
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        adlr_t = adlr_dict['data']['time']
        cas_t = cas_dict['data']['time']
        cip_t = cip_dict['data']['time']

        for offset in range(-9, 9):
            offset_cas_t = cas_t + offset
            #align all daxiets along time.set_aspect
            [adlr_inds, cip_inds, cas_inds] = match_multiple_arrays(( \
                                                np.around(adlr_t), \
                                                np.around(cip_t), \
                                                np.around(offset_cas_t)))
            cip_nconc, cas_nconc = get_nconc_data(adlr_dict, cas_dict, \
                                cip_dict, adlr_inds, cas_inds, cip_inds)

            no_nan_cip = np.logical_not(np.isnan(cip_nconc))
            no_nan_cas = np.logical_not(np.isnan(cas_nconc))
            no_nan_inds = np.logical_and(no_nan_cip, no_nan_cas)

            cip_nconc = cip_nconc[no_nan_inds]
            cas_nconc = cas_nconc[no_nan_inds]

            m, b, R, sig = linregress(cip_nconc, cas_nconc)
            print(offset, R**2, m)

def get_nconc_data(adlr_dict, cas_dict, cip_dict, adlr_inds, cas_inds, cip_inds):

    temp = adlr_dict['data']['temp'][adlr_inds]
    w = adlr_dict['data']['w'][adlr_inds]

    cas_t = cas_dict['data']['time'][cas_inds]
    nconc_cas = np.zeros(np.shape(cas_t))

    xi = cas_dict['data']['xi'][cas_inds]
    PAS = cas_dict['data']['PAS'][cas_inds]
    TAS = cas_dict['data']['TAS'][cas_inds]

    volume_corr_factor = xi/(PAS/TAS)

    for i in range(12, 17):
        var_name = 'nconc_' + str(i)
        nconc_i = cas_dict['data'][var_name][cas_inds]*volume_corr_factor
        nconc_cas += nconc_i

    nconc_cip = 0.5*cip_dict['data']['nconc_1'][cip_inds]

    return nconc_cas, nconc_cip 

if __name__ == "__main__":
    main()
