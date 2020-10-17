"""
perform and report chi squared test to judge statistical likelihood that
experimental ss_qss sampled distributions come from "true" distribution akin to
that reported from wrf simulation

all references herein to 'ss' are quasi-steady-state approximations to
supersaturation (for brevity)

also, 'exp' = 'experimental' and 'sim' = 'simulated'
"""
import numpy as np
from revhalo import BASE_DIR, DATA_DIR
from revhalo.ss_qss_calculations import get_lwc_from_cas, get_ss_vs_t_cas 

change_cas_corr = True
cutoff_bins = True
incl_rain = True
incl_vent = True
full_ss = True

lwc_filter_val = 1.e-4
w_cutoff = 2

def main():

    dict_dir = BASE_DIR + 'data/revmywrf/'
    ss_sim_dict = np.load(dict_dir + 'ss_sim_dict.npy', \
                            allow_pickle=True).item() 
    z_sim_dict = np.load(dict_dir + 'z_sim_dict.npy', \
                            allow_pickle=True).item() 

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_exp_alldates = None
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
        ss_exp = get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                    change_cas_corr, cutoff_bins, full_ss, \
                    incl_rain, incl_vent)

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        ss_exp = ss_exp[filter_inds]
        z_exp = z_exp[filter_inds]

        ss_exp_alldates = add_to_alldates_array(ss_exp, ss_exp_alldates)
        z_exp_alldates = add_to_alldates_array(z_exp, z_exp_alldates)

        print(date)
        #try:
        #    print_chi_squared_test_stats(ss_exp, z_exp, ss_sim_dict, z_sim_dict)
        #except ValueError:
        #    print('empty ss array for this date')
        #print()
        
    print('all dates')
    print_chi_squared_test_stats(ss_exp_alldates, z_exp_alldates, ss_sim_dict, z_sim_dict)
    print()

def print_chi_squared_test_stats(ss_exp, z_exp, ss_sim_dict, z_sim_dict):
    
    print(ss_sim_dict.keys())
    return
    for key in ss_sim_dict.keys():
        calc_and_print_chi_squared_test_stats_for_wrf_case(ss_exp, z_exp, \
                                        ss_sim_dict[key], z_sim_dict[key])

def calc_and_print_chi_squared_test_stats_for_wrf_case(ss_exp, z_exp, \
                                                        ss_sim, z_sim):

    n_ss_array = [10, 15, 20, 25, 30, 35] 
    n_z_array = [10, 20, 30] #fixed / manually alter for now

    for starting_n_ss in n_ss_array:
        for n_z in n_z_array:
            n_ss = starting_n_ss
            while True:
                (observed_ss_bin_counts, predicted_ss_bin_counts) = \
                    get_obs_and_pred_ss_distributions(n_ss, n_z, ss_exp, \
                                                      z_exp, ss_sim, z_sim)
                (is_rebinnable, tail_start_ind) = rebin_tail(predicted_ss_bin_counts)
                if is_rebinnable:
                    break
                else:
                    n_ss += 1

            n_dof = (tail_start_ind + 1) - 2 #2 constraints: n_ss, n_z
            chi_squared_no_tail = np.sum( \
                            (observed_ss_bin_counts[:tail_start_ind] \
                             - predicted_ss_bin_counts[:tail_start_ind])**2./ \
                             predicted_ss_bin_counts[:tail_start_ind])/n_dof
            chi_squared_tail = (np.sum(observed_ss_bin_counts[tail_start_ind:]) \
                                - np.sum(predicted_ss_bin_counts[tail_start_ind:]))**2./ \
                                np.sum(predicted_ss_bin_counts[tail_start_ind:])/n_dof

            print('num ss bins: ', tail_start_ind + 1)
            print('num z bins: ', n_z)
            print('chi sq val: ', chi_squared_no_tail + chi_squared_tail)
            print()

def get_obs_and_pred_ss_distributions(n_ss, n_z, ss_exp, \
                                      z_exp, ss_sim, z_sim):

    N_exp = np.shape(ss_exp)[0]

    ss_bins = get_bins(n_ss, ss_exp, ss_sim)
    z_bins = get_bins(n_z, z_exp, z_sim)
    
    pdf_exp = get_pdf(ss_bins, z_bins, ss_exp, z_exp)
    pdf_sim = get_pdf(ss_bins, z_bins, ss_sim, z_sim)

    #print(z_exp)
    #print(z_sim)
    #print(z_bins)
    #print(pdf_exp)
    #print(pdf_sim)

    observed_ss_bin_counts = np.array([np.sum(N_exp*pdf_exp[i, :]) \
                                        for i in range(n_ss)])

    #adjusted for different z distributions between exp and sim
    adjusted_pdf_sim = np.zeros(np.shape(pdf_sim))

    #summing quasi-manually bc numpy indexing syntax gives me anxiety
    for i in range(n_ss):
        for j in range(n_z):
            if np.sum(pdf_sim[:, j]) == 0:
                adjusted_pdf_sim[i, j] = 0 
            else:
                adjusted_pdf_sim[i, j] = \
                    np.sum(pdf_exp[:, j]*pdf_sim[i, j]/np.sum(pdf_sim[:, j]))

    predicted_ss_bin_counts = np.array([np.sum(N_exp*adjusted_pdf_sim[i, :]) \
                                        for i in range(n_ss)])

    print(observed_ss_bin_counts, predicted_ss_bin_counts)
    return (observed_ss_bin_counts, predicted_ss_bin_counts)

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

    note '1' and '2' designations are 'ss' and 'z'
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
