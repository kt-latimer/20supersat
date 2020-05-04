"""
Second round of HALO data processing: extract relevant data for ADLR, \
CAS, CDP, and NIXE-CAPS sets and convert to mks units. 

Input location: /data/halo/npy_raw
Output location: /data/halo/npy_proc
Output format: .npy file containing one dictionary formatted as: \
        {"dataset": HALO dataset number XXXX \
         "data": {"<clean var name 1>":<numpy array>, ...} \
         "units": {"<clean var name 1>":<string>, ...}}

update 1/20/20: also including LWC data in these files because it takes \
too long to run each time making figures. Format is dict['<data, units>']\
['lwc']['<0 or 1><0 or 1>'][<data>], where first boolean corresponds to 3um \
bin cutoff and second to changing CAS correction factor to match CDP; e.g. \
'01' means no bin cutoff but yes change CAS correction. There is also \
dict['data'] ['lwc_t_inds'], i.e. indices of dict['data']['time'] with which \
the lwc arrays are aligned (comes from having to match them with ADLR).
"""
from itertools import product
from os import listdir

import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import calc_lwc

input_data_dir =  DATA_DIR + 'npy_raw/'
output_data_dir = DATA_DIR + 'npy_proc/'

def main():
    """
    extract time, environment variables from ADLR; time, nconc, and \
    other available quantities from CAS, CDP, AND NIXE-CAPS. also \
    calculate lwc for CAS and CDP.
    """
    
    #clean variable names and their mks units (and scale factors into those \
    #units), as well as column indices of relevant values in the raw files.
    key_ind_dict = {'ADLR':\
                        {'var_names':['time', 'potl_temp', 'vert_wind_vel', \
                            'alt_asl', 'alt_pres', 'lat', 'long', 'stat_temp', \
                            'stat_pres', 'lwc', 'TAS', 'virt_potl_temp'], \
                        'var_units':['s', 'K', 'm/s', 'm', 'm', 'deg', 'deg', \
                            'K', 'Pa', 'g/g', 'm/s', 'K'], \
                        'var_inds':[0, 12, 17, 4, 5, 23, 24, 20, 7, 21, 9, 13], \
                        'var_scale':[1. for i in range(7)] \
                                    + [1., 100., 0.001, 1., 1.]}, \
                    'CAS':\
                        {'var_names':['time'] + ['nconc_'+str(i) for i in \
                            range(5, 17)] + ['nconc_tot_TAS_corr', \
                            'd_eff', 'd_vol', 'lwc_calc', 'PAS', 'TAS', 'xi'], \
                        'var_units': ['s'] + ['m^-3' for i in range(5, 18)] + \
                            ['m^-3', 'm', 'm', 'kg/m^3', 'm/s', 'm/s', 'none'], \
                        'var_inds':[i for i in range(20)], \
                        'var_scale':[1.] + [1.e6 for i in range(12)] + [1.e6, \
                            1.e-6, 1.e-6, 1.e3, 1., 1., 1.]}, \
                    'CDP':{\
                        'var_names':['time'] + ['nconc_'+str(i) for i in \
                            range(1, 16)] + ['d_geom'], \
                        'var_units':['s'] + ['m^-3' for i in \
                            range(1, 16)] + ['m'], \
                        'var_inds':[0] + [i for i in range(3, 18)] + [1], \
                        'var_scale':[1.] + [1.e6 for i in range(15)] + [1.e-6]}, \
                    'NIXECAPS':{\
                        'var_names':['time'] + ['nconc_'+str(i) for i in \
                            range(1, 72)] + ['PAS', 'TAS'], \
                        'var_units':['s'] + ['m^-3' for i in range(1, 72)] \
                            + ['m/s', 'm/s'], \
                        'var_inds':[0] + [i for i in range(9, 80)] + [3, 4], \
                        'var_scale':[1.] + [1.e6 for i in range(71)] + [1., 1.]}, \
                    'CLOUDFLAG':{\
                        'var_names':['time', 'in_cloud'], \
                        'var_units':['s', 'none'], \
                        'var_inds':[0, 1], \
                        'var_scale':[1., 1.]}, \
                    'SHARC':{\
                        'var_names':['time', 'abs_hum', 'Td', 'virt_potl_temp', \
                            'potl_temp', 'lwc', 'lwc_vol', 'RH_w', 'RH_i'], \
                        'var_units':['s', 'kg/m^3', 'K', 'K', 'K', 'g/g',
                            'ppmV', 'percent', 'percent'], \
                        'var_inds':[i for i in range(9)], \
                        'var_scale':[1., 0.001, 1., 1., 1., 0.001, 1., 1., 1.]}}
    
    #get names of data files with no issues (see notes)
    with open('good_ames_files.txt','r') as readFile:
        good_ames_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_ames_filenames:
        #pick out relevant datasets and load raw .npy files
        basename = filename[0:len(filename)-5]
        if 'sharc' in basename:
            setname = 'SHARC'
        elif 'adlr' in basename:
            setname = 'ADLR'
        #if 'adlr' in basename:
        #    setname = 'ADLR'
        #elif 'CAS_DPOL' in basename:
        #    setname = 'CAS'
        #    if basename[15:19] == '3914':
        #        #weird corrupted file.
        #        continue
        #elif 'CCP_CDP' in basename:
        #    setname = 'CDP'
        #elif 'NIXECAPS_AC' in basename:
        #    setname = 'NIXECAPS'
        #elif 'NIXECAPS_cloudflag' in basename:
        #    setname = 'CLOUDFLAG'
        else:
            continue     
        print(basename)
        raw_dict = np.load(input_data_dir+basename+'.npy', \
                allow_pickle=True).item()
        proc_dict = {'dataset':basename[15:19], \
                        'data':{}, \
                        'units':{}} 

        #select desired data from raw files and reformat slightly
        var_names = key_ind_dict[setname]['var_names']
        var_units = key_ind_dict[setname]['var_units']
        var_inds = key_ind_dict[setname]['var_inds']
        var_scale = key_ind_dict[setname]['var_scale']
        for i in range(len(var_names)):
            if setname == 'CDP' and i in range(1, 16):
            #need to divide ptcl num by sample volume
                proc_dict['data'][var_names[i]] = var_scale[i]\
                        *raw_dict['data'][:,var_inds[i]]/raw_dict['data'][:,2]
            else:
                proc_dict['data'][var_names[i]] = var_scale[i]\
                        *raw_dict['data'][:,var_inds[i]]
            proc_dict['units'][var_names[i]] = var_units[i]
        
        #save processed files
        datestr = raw_dict['flight_date'][0] + raw_dict['flight_date'][1] + \
                raw_dict['flight_date'][2]
        np.save(output_data_dir+setname+'_'+datestr, proc_dict)
    
    #if not modifying CAS/CDP files (else comment out line below)
    return

    #now calculate LWC values for CAS and CDP and add to raw files.
    files = [f for f in listdir(DATA_DIR + 'npy_proc/')]
    used_dates = []
    for f in files:
        #get flight date and check if already processed
        date = f[-12:-4]
        if date in used_dates:
            continue
        else:
            print(date)
            used_dates.append(date)
        
        #try to get adlr data for that date. if it doesn't exist, don't \
        #proceed because we won't have sufficient environmental data
        try:
            filename = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy' 
            adlrdata = np.load(filename, allow_pickle=True).item()
        except FileNotFoundError:
            adlrdata = {'data': None} 

        #process cas / cdp  datasets from flight date
        for setname in ['CAS', 'CDP']:
            try:    
                filename = DATA_DIR + 'npy_proc/' + setname + '_' \
                        + date + '.npy'
                dataset = np.load(filename, allow_pickle=True).item()
                updated_dataset = dataset.copy()
                updated_dataset['data']['lwc'] = {}
                updated_dataset['units']['lwc'] = {}
            except FileNotFoundError:
                print(filename + 'not found')
                continue
            
            #loop through all combos of booean params
            for cutoff_bins, change_cas_corr in product([True, False], repeat=2):
                booleankey = str(int(cutoff_bins)) \
                    + str(int(change_cas_corr)) 
                if setname=='CDP' and (booleankey == '10' or booleankey == '00'):
                    #avoid redundant calculations
                    pass
                else:
                    (lwc, t_inds) = calc_lwc(setname, dataset['data'], \
                        adlrdata['data'], cutoff_bins, change_cas_corr)
                #only this dataset is weird so fixing manually 
                if date == '20140906' and setname == 'CAS':
                    sorted_inds = np.argsort(t_inds)
                    lwc = lwc[sorted_inds]
                    t_inds = t_inds[sorted_inds]
                
                updated_dataset['data']['lwc'].update({booleankey: lwc})
                updated_dataset['units']['lwc'].update({booleankey: 'g/g'})
                
                #time inds don't depend on booleans so just update once.
                if booleankey == '00':
                    updated_dataset['data']['lwc_t_inds'] = t_inds
                    updated_dataset['units']['lwc_t_inds'] = 'none' 
            
            np.save(filename, updated_dataset)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
