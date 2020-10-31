"""
polish up 2011 caipeex data
"""
import numpy as np
import re

from rev11caipeex import DATA_DIR, CIP_bins, DMA_bins, FSSP_bins, PCASP_bins

var_info_dict = {'MET':\
                    {'var_names':['time', 'alt', 'temp', 'pres', \
                        'cloud_nconc_from_file', 'cloud_lwc_from_file', \
                        'pcasp_nconc_from_file'], \
                    'var_units':['sec', 'm', 'K', 'Pa', 'm^-3', \
                        'kg/m^3', 'm^-3'], \
                    'var_inds':[0, 1, 2, 4, 9, 11, 17], \
                    'var_scale':[1., 1., 1., 100., 1.e6, 1.e-3, 1.e6]}, \
                'FSSP':\
                    {'var_names':['time'] + ['nconc_'+str(i) for i in \
                        range(1, 31)], \
                    'var_units':['sec'] + ['m^-3' for i in range(30)], \
                    'var_inds':[0] + [i for i in range(2, 32)], \
                    'var_scale':[1.] + [1.e6 for i in range(30)]}, \
                'CIP':\
                    {'var_names':['time'] + ['nconc_'+str(i) for i in \
                        range(1, 36)], \
                    'var_units':['sec'] + ['m^-3' for i in range(35)], \
                    'var_inds':[0] + [i for i in range(2, 37)], \
                    'var_scale':[1.] + [1.e6 for i in range(35)]}, \
                'DMA':\
                    {'var_names':['time'] + ['nconc_'+str(i) for i in \
                        range(1, 61)], \
                    'var_units':['sec'] + ['m^-3' for i in range(60)], \
                    'var_inds':[0] + [i for i in range(1, 61)], \
                    'var_scale':[1.] + [1.7e4 for i in range(60)]}, \
                'PCASP':\
                    {'var_names':['time'] + ['nconc_'+str(i) for i in \
                        range(1, 29)], \
                    'var_units':['sec'] + ['m^-3' for i in range(28)], \
                    'var_inds':[0] + [i for i in range(1, 29)], \
                    'var_scale':[1.] + [1.e6 for i in range(28)]}}

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
        add_lwc_to_processed_dsd_files(date)

def make_processed_file_without_lwc(good_csv_filename):
    
    if 'CIP' in good_csv_filename:
        make_processed_cip_file(good_csv_filename)
    elif 'FSSP' in good_csv_filename:
        make_processed_fssp_file(good_csv_filename)
    elif 'PCASP' in good_csv_filename:
        make_processed_pcasp_file(good_csv_filename)
    elif 'DMA' in good_csv_filename:
        make_processed_dma_file(good_csv_filename)
    elif 'ALL' in good_csv_filename:
        make_processed_met_file(good_csv_filename)

def make_processed_cip_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date = raw_data_dict['date']
    var_inds = var_info_dict['CIP']['var_inds']
    var_names = var_info_dict['CIP']['var_names']
    var_scale = var_info_dict['CIP']['var_scale']
    var_units = var_info_dict['CIP']['var_units']
    processed_data_dict = {'setname': 'CIP', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_cip_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('CIP', date, processed_data_dict)

def make_processed_fssp_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date = raw_data_dict['date']
    var_inds = var_info_dict['FSSP']['var_inds']
    var_names = var_info_dict['FSSP']['var_names']
    var_scale = var_info_dict['FSSP']['var_scale']
    var_units = var_info_dict['FSSP']['var_units']
    processed_data_dict = {'setname': 'FSSP', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_fssp_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('FSSP', date, processed_data_dict)

def make_processed_pcasp_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date = raw_data_dict['date']
    var_inds = var_info_dict['PCASP']['var_inds']
    var_names = var_info_dict['PCASP']['var_names']
    var_scale = var_info_dict['PCASP']['var_scale']
    var_units = var_info_dict['PCASP']['var_units']
    processed_data_dict = {'setname': 'PCASP', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_pcasp_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('PCASP', date, processed_data_dict)

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

def make_processed_dma_file(good_csv_filename):

    good_numpy_filename = good_csv_filename[0:-3] + 'npy' #change file type
    raw_data_dict = np.load(DATA_DIR + 'npy_raw/' + good_numpy_filename, \
                            allow_pickle=True).item()
    data = raw_data_dict['data']
    date = raw_data_dict['date']
    var_inds = var_info_dict['DMA']['var_inds']
    var_names = var_info_dict['DMA']['var_names']
    var_scale = var_info_dict['DMA']['var_scale']
    var_units = var_info_dict['DMA']['var_units']
    processed_data_dict = {'setname': 'DMA', 'date': date, \
                           'raw_numpy_filename': good_numpy_filename, \
                           'data': {}, 'units': {}}

    for i, var_name in enumerate(var_names):
        processed_data_dict = add_var_to_processed_dma_dict(var_name, \
                                var_inds[i], var_scale[i], var_units[i], \
                                processed_data_dict, data)

    save_processed_file('DMA', date, processed_data_dict)
    

def add_var_to_processed_cip_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    #divide by 4 to average diams and get radii
    bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.
    
    diam_diffl_um = (CIP_bins['upper'] - CIP_bins['lower'])*1.e6

    if 'nconc' in var_name: #cip given as g water per m3 air per um 
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        bin_ind = nconc_ind - 1
        r = bin_radii[bin_ind]
        processed_data_dict['data'][var_name] = \
                data[var_ind]*var_scale*diam_diffl_um[bin_ind]/(rho_l*4./3.*np.pi*r**3.)
    else: #var is time.
        processed_data_dict['data'][var_name] = \
                                data[var_ind]*var_scale

    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_fssp_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    #divide by 4 to average diams and get radii
    bin_radii = (FSSP_bins['upper'] + FSSP_bins['lower'])/4.
    
    diam_diffl_um = (FSSP_bins['upper'] - FSSP_bins['lower'])*1.e6

    if 'nconc' in var_name: #fssp given as g water per m3 air per um 
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        bin_ind = nconc_ind - 1
        r = bin_radii[bin_ind]
        processed_data_dict['data'][var_name] = \
                data[var_ind]*var_scale*diam_diffl_um[bin_ind]/(rho_l*4./3.*np.pi*r**3.)
    else: #var is time.
        processed_data_dict['data'][var_name] = \
                                data[var_ind]*var_scale

    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_pcasp_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    bin_diams = (PCASP_bins['upper'] + PCASP_bins['lower'])/2.
    
    diam_diffl = (PCASP_bins['upper'] - PCASP_bins['lower'])

    if 'nconc' in var_name and 'pcasp' not in var_name: #pcasp data given as dN/dlogDp 
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        bin_ind = nconc_ind - 1
        other_var_name = 'dNdlogDp_' + str(nconc_ind) #jank af but oh well
        processed_data_dict['data'][var_name] = \
            data[:, var_ind]*var_scale*diam_diffl[bin_ind]/bin_diams[bin_ind]
        processed_data_dict['data'][other_var_name] = \
            data[:, var_ind]*var_scale
        processed_data_dict['units'][other_var_name] = var_unit
    else: #var is time
        processed_data_dict['data'][var_name] = \
                                data[:, var_ind]*var_scale

    processed_data_dict['units'][var_name] = var_unit

    return processed_data_dict

def add_var_to_processed_dma_dict(var_name, var_ind, var_scale, var_unit, \
                                    processed_data_dict, data):

    #dma data given as dN/dlogDp but bins are log scaled...Thara gave me \
    #this scale factor but I've had some issues figuring out how it's derived.
    processed_data_dict['data'][var_name] = \
                            data[:, var_ind]*var_scale
    processed_data_dict['units'][var_name] = var_unit

    if var_name != 'time':
        nconc_ind = int(re.findall(r'\d+', var_name)[0])
        other_var_name = 'dNdlogDp_' + str(nconc_ind) #jank af but oh well
        processed_data_dict['data'][other_var_name] = \
            data[:, var_ind]*1.e6
        processed_data_dict['units'][other_var_name] = var_unit

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

def add_lwc_to_processed_dsd_files(date):

    met_dict = np.load(output_data_dir + 'MET_' + date + '.npy', \
                allow_pickle=True).item()
    fssp_dict = np.load(output_data_dir + 'FSSP_' + date + '.npy', \
                allow_pickle=True).item()
    cip_dict = np.load(output_data_dir + 'CIP_' + date + '.npy', \
                allow_pickle=True).item()

    add_lwc_to_processed_fssp_file(date, met_dict, fssp_dict)
    add_lwc_to_processed_cip_file(date, met_dict, cip_dict)

def add_lwc_to_processed_fssp_file(date, met_dict, fssp_dict):
    
    #divide by 4 to average diams and get radii
    bin_radii = (FSSP_bins['upper'] + FSSP_bins['lower'])/4.
    
    met_t = met_dict['data']['time']
    fssp_t = fssp_dict['data']['time']

    met_t_inds = [t in fssp_t for t in met_t]
    
    #technically 25um is 24.6um for FSSP...
    r_cubed_sum_sub_5um_diam = np.zeros(np.shape(fssp_t))
    r_cubed_sum_5um_to_25um_diam = np.zeros(np.shape(fssp_t))
    r_cubed_sum_above_25um_diam = np.zeros(np.shape(fssp_t))

    for i in range(1, 3):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_sub_5um_diam += \
                fssp_dict['data'][var_key]*bin_radii[bin_ind]**3.

    for i in range(3, 15):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum_5um_to_25um_diam += \
                fssp_dict['data'][var_key]*bin_radii[bin_ind]**3.

    for i in range(15, 30):
        bin_ind = i - 1 
        var_key = 'nconc_' + str(i)
        r_cubed_sum_above_25um_diam += \
                fssp_dict['data'][var_key]*bin_radii[bin_ind]**3.

    rho_air = met_dict['data']['pres'][met_t_inds]/\
                (R_a*met_dict['data']['temp'][met_t_inds])

    fssp_dict['data']['lwc_sub_5um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_sub_5um_diam/rho_air
    fssp_dict['units']['lwc_sub_5um_diam'] = 'kg/kg'
    fssp_dict['data']['lwc_5um_to_25um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_5um_to_25um_diam/rho_air
    fssp_dict['units']['lwc_5um_to_25um_diam'] = 'kg/kg'
    fssp_dict['data']['lwc_above_25um_diam'] = \
            4./3.*np.pi*rho_l*r_cubed_sum_above_25um_diam/rho_air
    fssp_dict['units']['lwc_above_25um_diam'] = 'kg/kg'

    save_processed_file('FSSP', date, fssp_dict)

def add_lwc_to_processed_cip_file(date, met_dict, cip_dict):

    #divide by 4 to average diams and get radii
    bin_radii = (CIP_bins['upper'] + CIP_bins['lower'])/4.

    met_t = met_dict['data']['time']
    cip_t = cip_dict['data']['time']

    met_t_inds = [t in cip_t for t in met_t]
    
    r_cubed_sum = np.zeros(np.shape(cip_t))

    for i in range(1, 20):
        bin_ind = i - 1
        var_key = 'nconc_' + str(i)
        r_cubed_sum += cip_dict['data'][var_key]*bin_radii[bin_ind]**3.

    rho_air = met_dict['data']['pres'][met_t_inds]/\
                (R_a*met_dict['data']['temp'][met_t_inds])

    cip_dict['data']['lwc'] = 4./3.*np.pi*rho_l*r_cubed_sum/rho_air
    cip_dict['units']['lwc'] = 'kg/kg'

    save_processed_file('CIP', date, cip_dict)
                            
def save_processed_file(setname, date, processed_data_dict):

    processed_filename = setname + '_' + date + '.npy'
    np.save(output_data_dir + processed_filename, processed_data_dict)
                            
#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
