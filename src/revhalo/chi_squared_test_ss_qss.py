"""
perform and report chi squared test to judge statistical likelihood that
experimental ss_qss sampled distributions come from "true" distribution akin to
that reported from wrf simulation

all references herein to 'ss' are quasi-steady-state approximations to
supersaturation (for brevity)

also, 'exp' = 'experimental' and 'sim' = 'simulated'
"""
import numpy as np
from revhalo import DATA_DIR

def main():

    ss_sim_dict = {'Polluted': , 'Unpolluted':} 
    z_sim_dict = {'Polluted': , 'Unpolluted':} 

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_exp_alldates = None
    z_exp_alldates = None

    for date in good_dates:
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
        print_chi_squared_test_stats(ss_exp, z_exp, ss_sim_dict, z_sim_dict)
        print()
        
    print('all dates')
    print_chi_squared_test_stats(ss_exp_alldates, z_exp_alldates, ss_sim_dict, z_sim_dict)
    print()

def print_chi_squared_test_stats(ss_exp, z_exp, ss_sim_dict, z_sim_dict):
    
    for key in ss_sim_dict.keys():
       (n_ss, chi_squared) = get_chi_squared_test_stats(ss_exp, z_exp, \
                                                ss_sim_dict[key], z_sim_dict[key]) 

        print(key)
        print(n_ss, chi_squared)
        print()

def get_chi_squared_test_stats(ss_exp, z_exp, ss_sim, z_sim):

    n_ss = 4
    n_z = 30 #fixed / manually alter for now

    (observed_ss_bin_counts, predicted_ss_bin_counts) = \
            get_obs_and_pred_ss_distributions(n_ss, n_z, ss_exp, \
                                              z_exp, ss_sim, z_sim)

    while np.min(predicted_ss_bin_counts) > 5:
        n_ss += 1
        (observed_ss_bin_counts, predicted_ss_bin_counts) = \
                get_obs_and_pred_ss_distributions(n_ss, n_z, ss_exp, \
                                                  z_exp, ss_sim, z_sim)

    n_dof = np.shape(observed_ss_bin_counts)[0] - 2 #2 constraints: n_ss, n_z
    chi_squared = np.sum( \
                    (observed_ss_bin_counts - predicted_ss_bin_counts)**2./ \
                    predicted_ss_bin_counts)/n_dof

    return (n_ss, chi_squared)

def get_obs_and_pred_ss_distributions(n_ss, n_z, ss_exp, \
                                      z_exp, ss_sim, z_sim):

    N_exp = np.shape(ss_exp)[0]
    
    pdf_exp = get_pdf(n_ss, n_z, ss_exp, z_exp)
    pdf_sim = get_pdf(n_ss, n_z, ss_sim, z_sim)

    observed_ss_bin_counts = np.array([np.sum(N_exp*pdf_exp[i, :]) \
                                        for i in range(n_ss)])

    #adjusted for different z distributions between exp and sim
    adjusted_pdf_sim = np.zeros(np.shape(pdf_sim))

    #summing manually bc numpy indexing syntax gives me anxiety
    for i in range(n_ss):
        for j in range(n_z):
            for k in range(n_ss):
                for l in range(n_ss):
                    adjusted_pdf_sim[i, j] += \
                        pdf_exp[k, j]*pdf_sim[i, j]/pdf_sim[l, j]

    predicted_ss_bin_counts = np.array([np.sum(N_exp*adjusted_pdf_sim[i, :]) \
                                        for i in range(n_ss)])

    return (observed_ss_bin_counts, predicted_ss_bin_counts)

def get_pdf(n_1, n_2, var_1, var_2):
    """
    returns bivariate pdf normalized to 1 (i.e. sum of all entries = 1)
    """

    N = np.shape(var_1)[0]

    min_1 = np.min(var_1)
    max_1 = np.max(var_1)
    width_1 = (max_1 - min_1)/n_1

    min_2 = np.min(var_2)
    max_2 = np.max(var_2)
    width_2 = (max_2 - min_2)/n_2

    pdf = np.zeros((n1, n2))

    for i in range(n_1):
        for j in range(n_2):
            lo_var_1 = min_1 + i*width_1
            hi_var_1 = lo_var_1 + width_1
            lo_var_2 = min_2 + i*width_2
            hi_var_2 = lo_var_2 + width_2

            pdf[i, j] = np.sum(np.logical_and.reduce((
                                (var_1 >= lo_var_1), \
                                (var_1 < hi_var_1), \
                                (var_2 >= lo_var_2), \
                                (var_2 < hi_var_2))))

    #check for normalization
    if np.sum(pdf) != N:
        print('incorrect normalization!')
        print(np.sum(pdf), N)

    return pdf/N

if __name__ == "__main__":
    main()
