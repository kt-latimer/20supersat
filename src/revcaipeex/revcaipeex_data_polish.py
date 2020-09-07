"""
revisiting caipeex_data_polish to make the damn code legible 
"""
import numpy as np
import re

from revcaipeex import DATA_DIR, DSD_bins

var_info_dict = {'MET':\
                    {'var_names':['time', 'sectime', 'temp', 'pres', \
                        'alt', 'w', 'nconc_cdp', \
                        'reff_cdp', 'lwc_cdp'], 
                    'var_units':['HHMMSS', 'sec', 'K', 'Pa', 'm', 'm/s', \
                        'm^-3', 'm', 'kg/m^3'], \
                    'var_inds':[0, 1, 2, 4, 9, 14, 19, 20, 22], \
                    'var_scale':[1., 1., 1., 100., 1., 1., \
                        1.e6, 1.e-6, 1.e-3], \
                    'var_shift':[0., 0., 273., 0., 0., 0., 0., 0., 0.]}, \
                'DSD':\
                    {'var_names':['time'] + ['nconc_'+str(i) for i in \
                        range(1, 92)], \
                    'var_units':['HHMMSS'] + ['m^-3' for i in range(1, 92)], \
                    'var_inds':[0] + [i for i in range(9, 100)], \
                    'var_scale':[1.] + [1.e-3 for i in range(9, 100)], \
                    'var_shift':[0. for i in range(92)]}}

input_data_dir = DATA_DIR + 'npy_raw/'
output_data_dir = DATA_DIR + 'npy_proc/'

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of water (kg/m^3) b

def main():

    with open('good_csv_filenames.txt','r') as readFile:
        good_csv_filenames = [line.strip() for line in readFile.readlines()]

    for good_csv_filename in good_csv_filenames:
        make_processed_file_without_lwc(good_csv_filename)

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        print(date)
        add_lwc_to_processed_dsd_file(date)

def make_processed_file_without_lwc(good_csv_filename):
    
    if 'dsd' in good_csv_filename:
        make_processed_dsd_file(good_csv_filename)
    else:
        make_processed_met_file(good_csv_filename)

def make_processed_dsd_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
#    print(good_numpy_filename)
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
#    print(data)
    date = raw_data_dict['date']
    var_inds = var_info_dict['DSD']['var_inds']
    var_names = var_info_dict['DSD']['var_names']
    var_scale = var_info_dict['DSD']['var_scale']
    var_units = var_info_dict['DSD']['var_units']
    processed_data_dict = {'setname': 'DSD', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_dsd_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('DSD', date, processed_data_dict)

def make_processed_met_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date = raw_data_dict['date']
    var_inds = var_info_dict['MET']['var_inds']
    var_names = var_info_dict['MET']['var_names']
    var_scale = var_info_dict['MET']['var_scale']
    var_units = var_info_dict['MET']['var_units']
    processed_data_dict = {'setname': 'MET', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_met_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('MET', date, processed_data_dict)

def add_var_to_processed_dsd_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    #divide by 4 to average diams and get radii
    bin_radii = (DSD_bins['upper'] + DSD_bins['lower'])/4.

    if 'nconc' in var_name: #given as mass water per vol air 
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        bin_ind = nconc_ind - 1
                data[var_ind]*var_scale/(rho_l*4./3.*np.pi*r**3.)
    else: #var is time.
        processed_data_dict['data'][var_name] = \
                                data[var_ind]*var_scale

    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_met_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    if var_name == 'temp': #convert from C to K
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind] + 273.
    else:
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale

    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_lwc_to_processed_dsd_file(date):
    
    met_dict = np.load(output_data_dir + 'MET_' + date + '.npy', \
                allow_pickle=True).item()
    print(np.shape(met_dict['data']['time']))
    dsd_dict = np.load(output_data_dir + 'DSD_' + date + '.npy', \
                allow_pickle=True).item()
    print(np.shape(dsd_dict['data']['time']))

    #divide by 4 to average diams and get radii
    bin_radii = (DSD_bins['upper'] + DSD_bins['lower'])/4.
    
    dsd_t = dsd_dict['data']['time']
    
    r_cubed_sum_sub_5um_diam = np.zeros(np.shape(dsd_t))
    r_cubed_sum_5um_to_50um_diam = np.zeros(np.shape(dsd_t))
    r_cubed_sum_above_50um_diam = np.zeros(np.shape(dsd_t))

    for i in range(1, 4):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_sub_5um_diam += \
                dsd_dict['data'][var_key]*bin_radii[bin_ind]**3.

    for i in range(4, 31):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_5um_to_50um_diam += \
                dsd_dict['data'][var_key]*bin_radii[bin_ind]**3.

    for i in range(31, 92):
        bin_ind = i - 1 
        var_key = 'nconc_' + str(i)
        r_cubed_sum_above_50um_diam += \
                dsd_dict['data'][var_key]*bin_radii[bin_ind]**3.

    rho_air = met_dict['data']['pres']/(R_a*met_dict['data']['temp'])

    dsd_dict['data']['lwc_sub_5um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_sub_5um_diam/rho_air
    dsd_dict['units']['lwc_sub_5um_diam'] = 'kg/kg'
    dsd_dict['data']['lwc_5um_to_50um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_5um_to_50um_diam/rho_air
    dsd_dict['units']['lwc_5um_to_50um_diam'] = 'kg/kg'
    dsd_dict['data']['lwc_above_50um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_above_50um_diam/rho_air
    dsd_dict['units']['lwc_above_50um_diam'] = 'kg/kg'

    save_processed_file('DSD', date, dsd_dict) 

def save_processed_file(setname, date, processed_data_dict):

    processed_filename = setname + '_' + date + '.npy'
    np.save(output_data_dir + processed_filename, processed_data_dict)
                            
#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
