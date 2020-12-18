"""
Calculate linear regression parameters for CAS vs CDP in number concentration
and mean radius measured values. (Just prints everything out)
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from revhalo import DATA_DIR, FIG_DIR
from revhalo.utils import linregress, match_two_arrays

versionstr = 'v1_'

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        print(date)
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()

        adlr_t = adlr_dict['data']['time']
        cas_t = cas_dict['data']['time']

        for offset in range(-9, 9):
            offset_cas_t = cas_t + offset 
            #align all datasets along time.set_aspect
            [adlr_inds, cas_inds] = match_two_arrays(adlr_t, \
                                        np.around(offset_cas_t))
            adlr_tas = adlr_dict['data']['TAS'][adlr_inds]
            cas_tas = cas_dict['data']['TAS'][cas_inds]

            no_nan_adlr = np.logical_not(np.isnan(adlr_tas))
            no_nan_cas = np.logical_not(np.isnan(cas_tas))
            no_nan_inds = np.logical_and(no_nan_adlr, no_nan_cas)

            adlr_tas = adlr_tas[no_nan_inds]
            cas_tas = cas_tas[no_nan_inds]

            m_tas, b_tas, R_tas, sig_tas = linregress(adlr_tas, cas_tas)
            print(offset, R_tas**2, m_tas)
            
if __name__ == "__main__":
    main()
