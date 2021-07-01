"""
Look at alignment of ``xi`` variable for CAS and CIP based on time offset
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

change_CAS_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        print(date)
        ADLRfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        ADLR_dict = np.load(ADLRfile, allow_pickle=True).item()
        CASfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        CAS_dict = np.load(CASfile, allow_pickle=True).item()
        CIPfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        CIP_dict = np.load(CIPfile, allow_pickle=True).item()

        ADLR_t = ADLR_dict['data']['time']
        CAS_t = CAS_dict['data']['time']
        CIP_t = CIP_dict['data']['time']

        for offset in range(-9, 9):
            offset_CAS_t = CAS_t + offset
            #align all daxiets along time.set_aspect
            [ADLR_inds, CIP_inds, CAS_inds] = match_multiple_arrays(( \
                                                np.around(ADLR_t), \
                                                np.around(CIP_t), \
                                                np.around(offset_CAS_t)))

            CAS_xi = CAS_dict['data']['xi'][CAS_inds]
            CIP_xi = CAS_dict['data']['xi'][CIP_inds]

            no_nan_CIP = np.logical_not(np.isnan(CIP_xi))
            no_nan_CAS = np.logical_not(np.isnan(CAS_xi))
            no_nan_inds = np.logical_and(no_nan_CIP, no_nan_CAS)

            CIP_xi = CIP_xi[no_nan_inds]
            CAS_xi = CAS_xi[no_nan_inds]

            m, b, R, sig = linregress(CIP_xi, CAS_xi)
            print(offset, R**2, m)

if __name__ == "__main__":
    main()
