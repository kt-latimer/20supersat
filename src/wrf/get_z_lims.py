"""
get altitude ranges for both simulation cases within which WCUs constitute at
least 0.01% area fraction.
"""
import numpy as np

from wrf import DATA_DIR

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
area_frac_cutoff = 1.e-4

def main():

    ### wrf ###
    wrf_dict = np.load(DATA_DIR + 'filtered_data_dict.npy', \
                        allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        case_filtered_data_dict = wrf_dict[case_label]
        z = case_filtered_data_dict['z']
        z_bins = case_filtered_data_dict['z_bins']
        counts, z_bins = np.histogram(z, bins=z_bins, density=False)
        z_lim = get_z_lim(counts, z_bins)
        print(case_label)
        print(z_lim)

def get_z_lim(counts, z_bins):

    n_pts_tot = 450*450*84 #len*wid*time
    found_lo_bin = False
    bin_ind = 0

    while not found_lo_bin:
        if counts[bin_ind]/n_pts_tot >= 1.e-4:
            found_lo_bin = True
        bin_ind += 1

    lo_z_lim = z_bins[bin_ind]
    found_hi_bin = False

    while not found_hi_bin:
        if counts[bin_ind]/n_pts_tot < 1.e-4:
            found_hi_bin = True
        bin_ind += 1

    hi_z_lim = z_bins[bin_ind]

    return (lo_z_lim, hi_z_lim)

if __name__ == "__main__":
    main()
