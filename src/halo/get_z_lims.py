"""
get altitude range for both field campaign data from all dates combined
"""
import numpy as np

from halo import DATA_DIR
from halo.ss_functions import get_lwc_vs_t, get_ss_pred_vs_t, \
                              get_full_spectrum_dict

lwc_filter_val = 1.e-4
w_cutoff = 1

rmax = 102.e-6

change_CAS_corr = True
cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True

def main():
    
    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']
    z = ADLR_dict['data']['alt']
    ss_pred = get_ss_pred_vs_t(ADLR_dict, full_spectrum_dict, change_CAS_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    z = z[filter_inds]
    print(np.min(z), np.max(z))

if __name__ == "__main__":
    main()
