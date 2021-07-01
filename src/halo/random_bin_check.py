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

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    bin_nconc = np.array([])

    for date in good_dates:
        bin_nconc = np.concatenate(( \
            bin_nconc, get_bin_nconc_for_date(date, random_bin_ind)))

    print(random_bin_ind, HALO_bin_radii[random_bin_ind], \
        np.nanmean(bin_nconc)*HALO_bin_radii[random_bin_ind]/HALO_bin_dlogDp[random_bin_ind])
    
def get_bin_nconc_for_date(date, random_bin_ind):
    
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

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff_val), \
                            (temp > 273), \
                            (w > w_cutoff_val)))

    var_name = 'nconc_' + str(random_bin_ind+1) + '_corr'
    bin_nconc = get_nconc_contribution_from_spectrum_var(var_name, adlr_dict, \
                                    full_spectrum_dict, cutoff_bins, \
                                    incl_rain, HALO_bin_radii)

    return bin_nconc[filter_inds]

if __name__ == "__main__":
    main()
