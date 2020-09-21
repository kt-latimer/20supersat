import numpy as np

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

def get_sum_nconc_lo(dsd_input_vars):

    nconc_sum_lo = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i < 2.5e-6:
            nconc_var_name = 'nconc_' + str(i)
            nconc_sum_lo += dsd_input_vars[nconc_var_name][...]

    return nconc_sum_lo

def get_sum_rn_lo(dsd_input_vars):

    rn_sum_lo = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i < 2.5e-6:
            nconc_var_name = 'nconc_' + str(i)
            rn_sum_lo += r_i*dsd_input_vars[nconc_var_name][...]

    return rn_sum_lo

def get_sum_frn_lo(dsd_input_vars):

    frn_sum_lo = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i < 2.5e-6:
            nconc_var_name = 'nconc_' + str(i)
            f_var_name = 'f_' + str(i)
            frn_sum_lo += r_i*dsd_input_vars[nconc_var_name][...] \
                                *dsd_input_vars[f_var_name][...]

    return frn_sum_lo

def get_sum_nconc_mid(dsd_input_vars):

    nconc_sum_mid = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i >= 2.5e-6 and r_i < 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            nconc_sum_mid += dsd_input_vars[nconc_var_name][...]

    return nconc_sum_mid

def get_sum_rn_mid(dsd_input_vars):

    rn_sum_mid = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i >= 2.5e-6 and r_i < 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            rn_sum_mid += r_i*dsd_input_vars[nconc_var_name][...]

    return rn_sum_mid

def get_sum_frn_mid(dsd_input_vars):

    frn_sum_mid = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i >= 2.5e-6 and r_i < 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            f_var_name = 'f_' + str(i)
            frn_sum_mid += r_i*dsd_input_vars[nconc_var_name][...] \
                                *dsd_input_vars[f_var_name][...]

    return frn_sum_mid

def get_sum_nconc_hi(dsd_input_vars):

    nconc_sum_hi = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i > 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            nconc_sum_hi += dsd_input_vars[nconc_var_name][...]

    return nconc_sum_hi

def get_sum_rn_hi(dsd_input_vars):

    rn_sum_hi = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i > 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            rn_sum_hi += r_i*dsd_input_vars[nconc_var_name][...]

    return rn_sum_hi

def get_sum_frn_hi(dsd_input_vars):

    frn_sum_hi = np.zeros(np.shape(dsd_input_vars['nconc_1'][...))

    for i, r_i in enumerate(bin_radii):
        if r_i > 25e-6:
            nconc_var_name = 'nconc_' + str(i)
            f_var_name = 'f_' + str(i)
            frn_sum_hi += r_i*dsd_input_vars[nconc_var_name][...] \
                                *dsd_input_vars[f_var_name][...]

    return frn_sum_hi
