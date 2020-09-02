"""
revisiting halo_data_polish to figure out data discrepancies for lack of
documentation :'(
"""
import numpy as np
import re

from revhalo import DATA_DIR, CAS_bins, CDP_bins, CIP_bins

var_info_dict = {'ADLR':{'var_names':['time', 'pres', 'temp', 'w'], \
                         'var_inds':[0, 7, 10, 17], \
                         'var_units':['s', 'Pa', 'K', 'm/s'], \
                         'var_scale':[1., 100., 1., 1.]}, \
                 'CAS':{'var_names':['time'] + ['nconc_'+str(i) for i in \
                         range(5, 17)] + ['nconc_tot_from_file', \
                         'd_eff_from_file', 'PAS', 'TAS', 'xi'], \
                         'var_inds':[i for i in range(15)] + [17, 18, 19], \
                         'var_units':['s'] + ['m^-3' for i in range(5, 17)] \
                         + ['m^-3', 'm', 'm/s', 'm/s', 'None'], \
                         'var_scale':[1.] + [1.e6 for i in range(5, 17)] \
                         + [1.e6, 1.e-6, 1., 1., 1.]}, \
                 'CDP':{'var_names':['time', 'd_mean_from_file', \
                         'sample_vol'] + ['nconc_'+str(i) for i in \
                         range(1, 16)] + ['xi'], \
                         'var_inds':[i for i in range(19)], \
                         'var_units':['s', 'm', 'm^3'] + ['m^-3' for i in \
                         range(1, 16)] + ['None'], \
                         'var_scale':[1., 1.e-6, 1.e-6] + [1.e6 for i in \
                         range(1, 16)] + [1.]}, \
                 'CIP':{'var_names':['time', 'nconc_tot_from_file', \
                         'd_mean_from_file'] + ['nconc_'+str(i) for i in \
                         range(1, 20)] + ['xi'], \
                         'var_inds':[i for i in range(23)], \
                         'var_units':['s', 'm^-3', 'm'] + ['m^-3' for i in \
                         range(1, 20)] + ['None'], \
                         'var_scale':[1., 1.e6, 1.e-6] + [1.e6 for i in \
                         range(1, 20)] + [1.]}}

input_data_dir = DATA_DIR + 'npy_raw/'
output_data_dir = DATA_DIR + 'npy_proc/'

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of water (kg/m^3) 

def main():

    with open('good_ames_filenames.txt','r') as readFile:
        good_ames_filenames = [line.strip() for line in readFile.readlines()]

    for good_ames_filename in good_ames_filenames:
        make_processed_file_without_lwc(good_ames_filename)

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        print(date)
        add_lwc_to_processed_dsd_files(date)

def make_processed_file_without_lwc(good_ames_filename):
    
    if 'adlr' in good_ames_filename:
        make_processed_adlr_file(good_ames_filename)
    elif 'CAS_DPOL' in good_ames_filename:
        make_processed_cas_file_without_lwc(good_ames_filename)
    elif 'CDP' in good_ames_filename:
        make_processed_cdp_file_without_lwc(good_ames_filename)
    elif 'CIP' in good_ames_filename:
        make_processed_cip_file_without_lwc(good_ames_filename)

def make_processed_adlr_file(good_ames_filename):

    good_numpy_filename = good_ames_filename[0:-4] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date_array = raw_data_dict['date']
    date = date_array[0] + date_array[1] + date_array[2]
    var_inds = var_info_dict['ADLR']['var_inds']
    var_names = var_info_dict['ADLR']['var_names']
    var_scale = var_info_dict['ADLR']['var_scale']
    var_units = var_info_dict['ADLR']['var_units']
    processed_data_dict = {'setname': 'ADLR', 'date': raw_data_dict['date'], \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_adlr_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('ADLR', date, processed_data_dict)

def make_processed_cas_file_without_lwc(good_ames_filename):

    if '3914' in good_ames_filename:
        #weird corrupted file
        return

    good_numpy_filename = good_ames_filename[0:-4] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date_array = raw_data_dict['date']
    date = date_array[0] + date_array[1] + date_array[2]
    var_inds = var_info_dict['CAS']['var_inds']
    var_names = var_info_dict['CAS']['var_names']
    var_scale = var_info_dict['CAS']['var_scale']
    var_units = var_info_dict['CAS']['var_units']
    processed_data_dict = {'setname': 'CAS', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_cas_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    ##only this dataset is weird so fixing manually
    #if date == '20140906':
    #    sorted_inds = np.argsort(t_inds)
    #    lwc = lwc[sorted_inds]
    #    t_inds = t_inds[sorted_inds]

    save_processed_file('CAS', date, processed_data_dict)

def make_processed_cdp_file_without_lwc(good_ames_filename):

    good_numpy_filename = good_ames_filename[0:-4] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date_array = raw_data_dict['date']
    date = date_array[0] + date_array[1] + date_array[2]
    var_inds = var_info_dict['CDP']['var_inds']
    var_names = var_info_dict['CDP']['var_names']
    var_scale = var_info_dict['CDP']['var_scale']
    var_units = var_info_dict['CDP']['var_units']
    processed_data_dict = {'setname': 'CDP', 'date': raw_data_dict['date'], \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    #sample volume corrected to ambient conditions, 1/cm3
    sample_vol = data[:, var_inds[var_names.index('sample_vol')]]
    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_cdp_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                sample_vol, processed_data_dict, data)

    save_processed_file('CDP', date, processed_data_dict)

def make_processed_cip_file_without_lwc(good_ames_filename):

    good_numpy_filename = good_ames_filename[0:-4] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date_array = raw_data_dict['date']
    date = date_array[0] + date_array[1] + date_array[2]
    var_inds = var_info_dict['CIP']['var_inds']
    var_names = var_info_dict['CIP']['var_names']
    var_scale = var_info_dict['CIP']['var_scale']
    var_units = var_info_dict['CIP']['var_units']
    processed_data_dict = {'setname': 'CIP', 'date': raw_data_dict['date'], \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_cip_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('CIP', date, processed_data_dict)

def add_var_to_processed_adlr_dict(var_name, var_ind, var_scale, \
                                    var_unit, processed_data_dict, data):

    if var_name == 'time':
        processed_data_dict['data'][var_name] = \
                                np.around(data[:, var_ind])*var_scale
    else:
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale
    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_cas_dict(var_name, var_ind, var_scale, \
                                    var_unit, processed_data_dict, data):

    if var_name == 'time':
        processed_data_dict['data'][var_name] = \
                                np.around(data[:, var_ind])*var_scale
    else:
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale
    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_cdp_dict(var_name, var_ind, var_scale, \
                                  var_unit, sample_vol, processed_data_dict, data):

    if var_name == 'time':
        processed_data_dict['data'][var_name] = \
                                np.around(data[:, var_ind])*var_scale
    if 'nconc' in var_name: #given as number of ptcls in size bin
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale/sample_vol
    else:
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale
    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_cip_dict(var_name, var_ind, var_scale, \
                                    var_unit, processed_data_dict, data):

    if var_name == 'time':
        processed_data_dict['data'][var_name] = \
                                np.around(data[:, var_ind])*var_scale
    if 'nconc' in var_name and 'tot' not in var_name: #given as dN/dlogDp 
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        bin_ind = nconc_ind - 1
        dlogDp = np.log10(CIP_bins['upper'][bin_ind]/CIP_bins['lower'][bin_ind])
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale*dlogDp
    else:
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale
    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_lwc_to_processed_dsd_files(date):

    adlr_dict = np.load(output_data_dir + 'ADLR_' + date + '.npy', \
                allow_pickle=True).item()
    cas_dict = np.load(output_data_dir + 'CAS_' + date + '.npy', \
                allow_pickle=True).item()
    cdp_dict = np.load(output_data_dir + 'CDP_' + date + '.npy', \
                allow_pickle=True).item()
    cip_dict = np.load(output_data_dir + 'CIP_' + date + '.npy', \
                allow_pickle=True).item()

    (adlr_dict, cas_dict, cdp_dict, cip_dict) = \
            sync_and_match_times(adlr_dict, cas_dict, cdp_dict, cip_dict)

    add_lwc_to_processed_cas_file(date, adlr_dict, cas_dict)
    add_lwc_to_processed_cdp_file(date, adlr_dict, cdp_dict)
    add_lwc_to_processed_cip_file(date, adlr_dict, cip_dict)

def sync_and_match_times(adlr_dict, cas_dict, cdp_dict, cip_dict):

    delta_cas = get_adlr_cas_offset(adlr_dict, cas_dict)
    delta_cdp = get_cas_cdp_offset(cas_dict, cdp_dict) + delta_cas
    delta_cip = delta_cdp #cdp and cip are already synced (I checked
                          #this manually via xi values in both files)
    
    adlr_t = adlr_dict['data']['time']
    cas_t = cas_dict['data']['time'] + delta_cas
    cdp_t = cdp_dict['data']['time'] + delta_cdp
    cip_t = cip_dict['data']['time'] + delta_cip

    cas_dict['data']['time'] = cas_t
    cdp_dict['data']['time'] = cdp_t
    cip_dict['data']['time'] = cip_t

    [adlr_inds, cas_inds, cdp_inds, cip_inds] = \
                match_multiple_arrays([adlr_t, cas_t, cdp_t, cip_t])

    adlr_dict['data'] = get_time_matched_data_dict(adlr_dict['data'], adlr_inds) 
    cas_dict['data'] = get_time_matched_data_dict(cas_dict['data'], cas_inds) 
    cdp_dict['data'] = get_time_matched_data_dict(cdp_dict['data'], cdp_inds) 
    cip_dict['data'] = get_time_matched_data_dict(cip_dict['data'], cip_inds) 

    return (adlr_dict, cas_dict, cdp_dict, cip_dict)

def get_adlr_cas_offset(adlr_dict, cas_dict):

    #some day do some stuff
    return 0

def get_cas_cdp_offset(cas_dict, cdp_dict):

    #some day do some stuff
    return 0

def match_multiple_arrays(arrays):

    """
    Return: [inds1, ... , indsN] where arr1[inds1] = ... = arrN[indsN].
    Assumes all arrays are sorted in the same order (ie time series)
    probably a better way to do this recursively but I never learned that shit xd
    """
    inds = [[i for i in range(len(arrays[0]))]]
    for i, array in enumerate(arrays[:-1]):
        (inds1, inds2) = match_two_arrays([array[i] for i in inds[-1]], arrays[i+1])
        inds = [[indsj[i] for i in inds1] for indsj in inds]
        inds.append(inds2)
    return [np.array(indsarr) for indsarr in inds]

def match_two_arrays(arr1, arr2):

    """
    Return: (inds1, inds2) where arr1[inds1] = arr2[inds2].
    Assumes arr1 and arr2 are both sorted in the same order (ie time series)
    """
    inds1 = []
    inds2 = []
    startind2 = 0
    for i1, x1 in enumerate(arr1):
        for i2, x2 in enumerate(arr2[startind2:]):
            if x1 == x2:
                inds1.append(i1)
                inds2.append(i2+startind2)
                startind2 = i2 + startind2 + 1
                break
    return(inds1, inds2)

def get_time_matched_data_dict(data_dict, inds):

    for key in data_dict.keys():
        data_dict[key] = data_dict[key][inds]

    return data_dict

def add_lwc_to_processed_cas_file(date, adlr_dict, cas_dict):

    cas_dict = add_corrected_nconc_to_processed_cas_file(adlr_dict, cas_dict)
    
    #divide by 4 to average diams and get radii
    bin_radii = (CAS_bins['upper'] + CAS_bins['lower'])/4.

    adlr_t = adlr_dict['data']['time']
    
    r_cubed_sum_sub_5um_diam = np.zeros(np.shape(adlr_t))
    r_cubed_sum_5um_to_25um_diam = np.zeros(np.shape(adlr_t))
    r_cubed_sum_above_25um_diam = np.zeros(np.shape(adlr_t))

    r_cubed_sum_sub_5um_diam_corr = np.zeros(np.shape(adlr_t))
    r_cubed_sum_5um_to_25um_diam_corr = np.zeros(np.shape(adlr_t))
    r_cubed_sum_above_25um_diam_corr = np.zeros(np.shape(adlr_t))

    for i in range(5, 8):
        bin_ind = i - 5
        var_key = 'nconc_' + str(i)
        r_cubed_sum_sub_5um_diam += \
                cas_dict['data'][var_key]*bin_radii[bin_ind]**3.
        r_cubed_sum_sub_5um_diam_corr += \
                cas_dict['data'][var_key+'_corr']*bin_radii[bin_ind]**3.

    for i in range(8, 12):
        bin_ind = i - 5
        var_key = 'nconc_' + str(i)
        r_cubed_sum_5um_to_25um_diam += \
                cas_dict['data'][var_key]*bin_radii[bin_ind]**3.
        r_cubed_sum_5um_to_25um_diam_corr += \
                cas_dict['data'][var_key+'_corr']*bin_radii[bin_ind]**3.

    for i in range(12, 17):
        bin_ind = i - 5
        var_key = 'nconc_' + str(i)
        r_cubed_sum_above_25um_diam += \
                cas_dict['data'][var_key]*bin_radii[bin_ind]**3.
        r_cubed_sum_above_25um_diam_corr += \
                cas_dict['data'][var_key+'_corr']*bin_radii[bin_ind]**3.

    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    cas_dict['data']['lwc_sub_5um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_sub_5um_diam/rho_air
    cas_dict['units']['lwc_sub_5um_diam'] = 'kg/kg'
    cas_dict['data']['lwc_5um_to_25um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_5um_to_25um_diam/rho_air
    cas_dict['units']['lwc_5um_to_25um_diam'] = 'kg/kg'
    cas_dict['data']['lwc_above_25um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_above_25um_diam/rho_air
    cas_dict['units']['lwc_above_25um_diam'] = 'kg/kg'

    cas_dict['data']['lwc_sub_5um_diam_corr'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_sub_5um_diam_corr/rho_air
    cas_dict['units']['lwc_sub_5um_diam_corr'] = 'kg/kg'
    cas_dict['data']['lwc_5um_to_25um_diam_corr'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_5um_to_25um_diam_corr/rho_air
    cas_dict['units']['lwc_5um_to_25um_diam_corr'] = 'kg/kg'
    cas_dict['data']['lwc_above_25um_diam_corr'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_above_25um_diam_corr/rho_air
    cas_dict['units']['lwc_above_25um_diam_corr'] = 'kg/kg'

    save_processed_file('CAS', date, cas_dict)

def add_lwc_to_processed_cdp_file(date, adlr_dict, cdp_dict):
    
    #divide by 4 to average diams and get radii
    bin_radii = (CDP_bins['upper'] + CDP_bins['lower'])/4.
    
    adlr_t = adlr_dict['data']['time']
    
    r_cubed_sum_sub_5um_diam = np.zeros(np.shape(adlr_t))
    r_cubed_sum_above_5um_diam = np.zeros(np.shape(adlr_t))

    for i in range(1, 3):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_sub_5um_diam += \
                cdp_dict['data'][var_key]*bin_radii[bin_ind]**3.

    for i in range(3, 16):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_above_5um_diam += \
                cdp_dict['data'][var_key]*bin_radii[bin_ind]**3.

    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    cdp_dict['data']['lwc_sub_5um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_sub_5um_diam/rho_air
    cdp_dict['units']['lwc_sub_5um_diam'] = 'kg/kg'
    cdp_dict['data']['lwc_above_5um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_above_5um_diam/rho_air
    cdp_dict['units']['lwc_above_5um_diam'] = 'kg/kg'

    save_processed_file('CDP', date, cdp_dict)

def add_lwc_to_processed_cip_file(date, adlr_dict, cip_dict):

    #divide by 4 to average diams and get radii
    bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

    adlr_t = adlr_dict['data']['time']
    
    r_cubed_sum = np.zeros(np.shape(adlr_t))

    for i in range(1, 20):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum += cip_dict['data'][var_key]*bin_radii[bin_ind]**3.

    rho_air = adlr_dict['data']['pres']/(R_a*adlr_dict['data']['temp'])

    cip_dict['data']['lwc'] = 4./3.*np.pi*rho_l*r_cubed_sum/rho_air
    cip_dict['units']['lwc'] = 'kg/kg'

    save_processed_file('CIP', date, cip_dict)

def add_corrected_nconc_to_processed_cas_file(adlr_dict, cas_dict):

    xi = cas_dict['data']['xi']
    PAS = cas_dict['data']['PAS']
    TAS = cas_dict['data']['TAS']

    volume_corr_factor = xi/(PAS/TAS)

    for i in range(5, 17):
        var_key = 'nconc_' + str(i)
        cas_dict['data'][var_key+'_corr'] = \
                cas_dict['data'][var_key]*volume_corr_factor
        cas_dict['units'][var_key+'_corr'] = 'm^-3'

    return cas_dict

def save_processed_file(setname, date, processed_data_dict):

    processed_filename = setname + '_' + date + '.npy'
    np.save(output_data_dir + processed_filename, processed_data_dict)
                            
#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
