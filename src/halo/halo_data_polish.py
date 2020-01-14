"""
Second round of HALO data processing: extract relevant data for ADLR, \
CAS, CDP, and NIXE-CAPS sets and convert to mks units. 

Input location: /data/halo/npy_raw
Output location: /data/halo/npy_proc
Output format: .npy file containing one dictionary formatted as: \
        {"dataset": HALO dataset number XXXX \
         "data": {"<clean var name 1>":<numpy array>, ...} \
         "units": {"<clean var name 1>":<string>, ...}}
"""
import numpy as np

input_data_dir =  '/home/klatimer/proj/20supersat/data/halo/npy_raw/'
output_data_dir = '/home/klatimer/proj/20supersat/data/halo/npy_proc/'

def main():
    """
    extract time, environment variables from ADLR; time, nconc, and \
    other available quantities from CAS, CDP, AND NIXE-CAPS.
    """
    
    #clean variable names and their mks units (and scale factors into those \
    #units), as well as column indices of relevant values in the raw files.
    key_ind_dict = {'ADLR':\
                        {'var_names':['time', 'potl_temp', 'vert_wind_vel', \
                            'alt_asl', 'alt_pres', 'lat', 'long'], \
                        'var_units':['s', 'K', 'm/s', 'm', 'm', 'deg', 'deg'], \
                        'var_inds':[0, 11, 16, 3, 4, 22, 23], \
                        'var_scale':[1. for i in range(7)]}, \
                    'CAS':\
                        {'var_names':['time'] + ['nconc_'+str(i) for i in \
                            range(5, 17)] + ['nconc_tot_TAS_corr', 'd_eff', \
                            'd_vol', 'PAS', 'TAS', 'xi'], \
                        'var_units': ['s'] + ['m^-3' for i in range(5, 18)] + \
                            ['m', 'm', 'm/s', 'm/s', 'none'], \
                        'var_inds':[i for i in range(19)], \
                        'var_scale':[1.] + [1.e6 for i in range(13)] + [1.e-6,
                            1.e-6, 1., 1., 1., 1., 1.]}, \
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
                        'var_scale':[1., 1.]}}
    
    #get names of data files with no issues (see notes)
    with open('good_ames_files.txt','r') as readFile:
        good_ames_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_ames_filenames:
        #pick out relevant datasets and load raw .npy files
        basename = filename[0:len(filename)-5]
        if 'adlr' in basename:
            setname = 'ADLR'
        elif 'CAS_DPOL' in basename:
            setname = 'CAS'
            if basename[15:19] == '3914':
                #weird corrupted file.
                continue
        elif 'CCP_CDP' in basename:
            setname = 'CDP'
        elif 'NIXECAPS_AC' in basename:
            setname = 'NIXECAPS'
        elif 'NIXECAPS_cloudflag' in basename:
            setname = 'CLOUDFLAG'
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

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
