"""
perform and report chi squared test to judge statistical likelihood that
experimental nconc sampled distributions come from "true" distribution akin to
that reported from wrf simulation

'exp' = 'experimental' and 'sim' = 'simulated'
"""
import numpy as np
from revhalo import BASE_DIR, DATA_DIR
from revhalo.ss_qss_calculations import get_lwc_from_cas, \
                                    get_nconc_vs_t_from_cas_and_cip

change_cas_corr = True
cutoff_bins = True
incl_rain = True
incl_vent = True
full_ss = True

lwc_filter_val = 1.e-4
w_cutoff = 2

def main():

    dict_dir = BASE_DIR + 'data/revmywrf/'
    nconc_sim_dict = np.load(dict_dir + 'v1_nconc_sim_dict.npy', \
                            allow_pickle=True).item() 
    z_sim_dict = np.load(dict_dir + 'v1_z_sim_dict.npy', \
                            allow_pickle=True).item() 

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    nconc_exp_alldates = None
    z_exp_alldates = None

    for date in good_dates:#[0:1]:
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
        cip_dict = np.load(cipfile, allow_pickle=True).item()

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        w = adlr_dict['data']['w']
        z_exp = adlr_dict['data']['alt']
        #remove "_and_cip" if not inclrain below
        nconc_exp = get_nconc_vs_t_from_cas_and_cip(adlr_dict, cas_dict, \
                            cip_dict, change_cas_corr, cutoff_bins, \
                            incl_rain, incl_vent)

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        nconc_exp = nconc_exp[filter_inds]
        z_exp = z_exp[filter_inds]

        nconc_exp_alldates = add_to_alldates_array(nconc_exp, nconc_exp_alldates)
        z_exp_alldates = add_to_alldates_array(z_exp, z_exp_alldates)

        print(date)
        #try:
        #    print_chi_squared_test_stats(nconc_exp, z_exp, nconc_sim_dict, z_sim_dict)
        #except ValueError:
        #    print('empty ss array for this date')
        #print()
        
    print('all dates')
    print_chi_squared_test_stats(nconc_exp_alldates, z_exp_alldates, nconc_sim_dict, z_sim_dict)
    print()

def print_chi_squared_test_stats(nconc_exp, z_exp, nconc_sim_dict, z_sim_dict):
    
    n_nconc_array = []
    n_z_array = []
    chi_sq_array = []

    for key in nconc_sim_dict.keys():
        (partial_n_nconc_array, partial_n_z_array, partial_chi_sq_array) = \
            calc_and_print_chi_squared_test_stats_for_wrf_case(nconc_exp, z_exp, \
                                        nconc_sim_dict[key], z_sim_dict[key])

        n_nconc_array += partial_n_nconc_array
        n_z_array += partial_n_z_array
        chi_sq_array += partial_chi_sq_array

    for i in range(len(n_nconc_array)):
        print(n_nconc_array[i], n_z_array[i], chi_sq_array[i])

def calc_and_print_chi_squared_test_stats_for_wrf_case(nconc_exp, z_exp, \
                                                        nconc_sim, z_sim):

    starting_n_nconc_array = [325] 
    starting_n_z_array = [10, 20, 30] #fixed / manually alter for now

    n_nconc_array = []
    n_z_array = []
    chi_sq_array = []

    for starting_n_nconc in starting_n_nconc_array:
        for n_z in starting_n_z_array:
            (observed_nconc_bin_counts, predicted_nconc_bin_counts) = \
                get_obs_and_pred_nconc_distributions(starting_n_nconc, n_z, nconc_exp, \
                                                  z_exp, nconc_sim, z_sim)
            if observed_nconc_bin_counts is not None:
                n_nconc = np.shape(observed_nconc_bin_counts)[0]
                n_dof = n_nconc - 2 #2 constraints: n_nconc, n_z
                chi_squared = np.sum(
                    (observed_nconc_bin_counts - predicted_nconc_bin_counts)**2./ \
                    predicted_nconc_bin_counts)/n_dof

            n_nconc_array.append(n_nconc)
            n_z_array.append(n_z)
            chi_sq_array.append(chi_squared)

    return (n_nconc_array, n_z_array, chi_sq_array)

def get_obs_and_pred_nconc_distributions(starting_n_nconc, n_z, nconc_exp, \
                                      z_exp, nconc_sim, z_sim):

    N_exp = np.shape(nconc_exp)[0]

    nconc_bins = get_bins(starting_n_nconc, nconc_exp, nconc_sim)
    z_bins = get_bins(n_z, z_exp, z_sim)
    
    pdf_exp = get_pdf(nconc_bins, z_bins, nconc_exp, z_exp)
    pdf_sim = get_pdf(nconc_bins, z_bins, nconc_sim, z_sim)

    #print(z_exp)
    #print(z_sim)
    #print(z_bins)
    #print(pdf_exp)
    #print(pdf_sim)

    observed_nconc_bin_counts = np.array([np.sum(N_exp*pdf_exp[i, :]) \
                                        for i in range(starting_n_nconc)])

    predicted_nconc_bin_counts = get_adjusted_pred_nconc_bin_counts(pdf_exp, \
                                        pdf_sim, starting_n_nconc, n_z, N_exp)

    (observed_nconc_bin_counts, predicted_nconc_bin_counts) = \
        rebin_nconc_bin_counts(observed_nconc_bin_counts, predicted_nconc_bin_counts)

    print(observed_nconc_bin_counts, predicted_nconc_bin_counts)
    return (observed_nconc_bin_counts, predicted_nconc_bin_counts)

def get_adjusted_pred_nconc_bin_counts(pdf_exp, pdf_sim, starting_n_nconc, n_z, N_exp):

    #adjusted for different z distributions between exp and sim
    adjusted_pdf_sim = np.zeros(np.shape(pdf_sim))

    #summing quasi-manually bc numpy indexing syntax gives me anxiety
    for i in range(starting_n_nconc):
        for j in range(n_z):
            if np.sum(pdf_sim[:, j]) == 0:
                adjusted_pdf_sim[i, j] = 0 
            else:
                adjusted_pdf_sim[i, j] = \
                    np.sum(pdf_exp[:, j]*pdf_sim[i, j]/np.sum(pdf_sim[:, j]))

    predicted_nconc_bin_counts = np.array([np.sum(N_exp*adjusted_pdf_sim[i, :]) \
                                        for i in range(starting_n_nconc)])

    return predicted_nconc_bin_counts

def rebin_nconc_bin_counts(observed_nconc_bin_counts, predicted_nconc_bin_counts):

    print(predicted_nconc_bin_counts)
    new_observed_nconc_bin_counts = []
    new_predicted_nconc_bin_counts = []

    current_bin_end_ind = np.shape(observed_nconc_bin_counts)[0]
    bin_sum = 0
    current_ind = current_bin_end_ind - 1

    while current_ind >= 0:
        bin_sum += predicted_nconc_bin_counts[current_ind]
        if bin_sum >= 5:
            new_predicted_nconc_bin_counts.insert(0, np.sum( \
                predicted_nconc_bin_counts[current_ind:current_bin_end_ind]))
            new_observed_nconc_bin_counts.insert(0, np.sum( \
                observed_nconc_bin_counts[current_ind:current_bin_end_ind]))
            current_bin_end_ind = current_ind
            bin_sum = 0
        elif current_ind == 0:
            new_predicted_nconc_bin_counts[0] += np.sum( \
                predicted_nconc_bin_counts[current_ind:current_bin_end_ind])
            new_observed_nconc_bin_counts[0] += np.sum( \
                observed_nconc_bin_counts[current_ind:current_bin_end_ind])
        current_ind -= 1

    if len(new_predicted_nconc_bin_counts) >= 4:
        return (np.array(new_observed_nconc_bin_counts), \
                np.array(new_predicted_nconc_bin_counts))
    else:
        print(len(new_predicted_nconc_bin_counts))
        return (None, None)

def get_bins(n_bins, var_1, var_2):
    """
    returns bins commensurate with both datasets

    note '1' and '2' designations are 'exp' and 'sim'
    """

    min_1 = np.min(var_1)
    max_1 = 1.001*(np.max(var_1)) #so last bin includes max

    min_2 = np.min(var_2)
    max_2 = 1.001*(np.max(var_2)) #so last bin includes max

    common_min = np.min([min_1, min_2])
    common_max = np.max([max_1, max_2])

    bin_width = (common_max - common_min)/n_bins

    bins = np.array([common_min + i*bin_width for i in range(n_bins + 1)])

    return bins

def get_pdf(bins_1, bins_2, var_1, var_2):
    """
    returns bivariate pdf normalized to 1 (i.e. sum of all entries = 1)

    note '1' and '2' designations are 'nconc' and 'z'
    """

    N = np.shape(var_1)[0]
    n_1 = np.shape(bins_1)[0] - 1
    n_2 = np.shape(bins_2)[0] - 1

    pdf = np.zeros((n_1, n_2))

    for i, lo_val_1 in enumerate(bins_1[:-1]):
        for j, lo_val_2 in enumerate(bins_2[:-1]):
            hi_val_1 = bins_1[i+1] 
            hi_val_2 = bins_2[j+1] 

            pdf[i, j] = np.sum(np.logical_and.reduce((
                                (var_1 >= lo_val_1), \
                                (var_1 < hi_val_1), \
                                (var_2 >= lo_val_2), \
                                (var_2 < hi_val_2))))

    #check for normalization
    if np.sum(pdf) != N:
        print('incorrect normalization!')
        print(np.sum(pdf), N)

    return pdf/N

def rebin_tail(pdf):
    
    n_bins = np.shape(pdf)[0]
    tail_sum = 0
    for i in range(n_bins):
        tail_sum += pdf[n_bins - 1 - i]
        if tail_sum < 5 and i == n_bins - 4:
            return (False, None)
        elif tail_sum >= 5:
            if pdf[n_bins - 2 - i] < 5:
                print('bump in rebinned pdf')
            return (True, n_bins - 1 - i) 

def add_to_alldates_array(arr, arr_alldates):

    if arr_alldates is None:
        return arr 
    else:
        return np.concatenate((arr_alldates, arr))

if __name__ == "__main__":
    main()
