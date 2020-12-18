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
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        cas_t = cas_dict['data']['time']
        cip_t = cip_dict['data']['time']

        for offset in range(-9, 9):
            offset_cas_t = cas_t + offset 
            #align all daxiets along time.set_aspect
            [cip_inds, cas_inds] = match_two_arrays(cip_t, \
                                        np.around(offset_cas_t))
            cip_xi = cip_dict['data']['xi'][cip_inds]
            cas_xi = cas_dict['data']['xi'][cas_inds]

            no_nan_cip = np.logical_not(np.isnan(cip_xi))
            no_nan_cas = np.logical_not(np.isnan(cas_xi))
            no_nan_inds = np.logical_and(no_nan_cip, no_nan_cas)

            cip_xi = cip_xi[no_nan_inds]
            cas_xi = cas_xi[no_nan_inds]

            m_xi, b_xi, R_xi, sig_xi = linregress(cip_xi, cas_xi)
            print(offset, R_xi**2, m_xi)
            
if __name__ == "__main__":
    main()
