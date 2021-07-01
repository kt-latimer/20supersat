"""
check d(N*r)/dlogDp for a random HALO bin
"""
import numpy as np

from halo import CAS_bins, CIP_bins, DATA_DIR
from ss_functions import get_lwc_vs_t, \
                         get_nconc_contribution_from_spectrum_var, \
                         get_full_spectrum_bin_radii, \
                         get_full_spectrum_dict, \
                         get_full_spectrum_dlogDp

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = True 

HALO_bin_radii = get_full_spectrum_bin_radii(CAS_bins, CIP_bins, 'log') 
HALO_bin_dlogDp = get_full_spectrum_dlogDp(CAS_bins, CIP_bins)

lwc_cutoff_val = 1.e-4
w_cutoff_val = 1

rmax = 102.e-6

def main():
    
    random_bin_ind = np.random.randint(0, np.shape(HALO_bin_radii)[0])

    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    bin_nconc = get_bin_nconc(random_bin_ind, spectrum_dict)

    print(random_bin_ind, HALO_bin_radii[random_bin_ind], \
        np.nanmean(bin_nconc)*HALO_bin_radii[random_bin_ind]/HALO_bin_dlogDp[random_bin_ind])
    
def get_bin_nconc(random_bin_ind, ADLR_dict, spectrum_dict):

    lwc = get_lwc_vs_t(ADLR_dict, spectrum_dict, cutoff_bins, rmax)
    temp = ADLR_dict['data']['temp']
    w = ADLR_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (w > w_cutoff_val)))

    var_name = 'nconc_' + str(random_bin_ind+1) + '_corr'
    bin_nconc = get_nconc_contribution_from_spectrum_var(var_name, ADLR_dict, \
                                    spectrum_dict, cutoff_bins, \
                                    incl_rain, HALO_bin_radii)

    return bin_nconc[filter_inds]

if __name__ == "__main__":
    main()
