"""
Second round of HALO data processing: extract relevant data for ADLR, \
CAS, CDP, NIXE-CAPS, SHARC, and CIP sets and convert to mks units. 

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

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
#from caipeex.utils import centr_dsd, DSD_bins

input_data_dir =  DATA_DIR + 'npy_raw/'
output_data_dir = DATA_DIR + 'npy_proc/'
#centr_dsd = DSD_bins['lower']/2.
#centr_dsd = DSD_bins['lower']

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

def main():
    """
    extract time, environment variables from ADLR; time, nconc, and \
    other available quantities from CAS, CDP, NIXE-CAPS, SHARC, and 
    CIP. also calculate lwc for CAS and CDP.
    """
    
    #centr_dsd = centr_dsd*2.
    from caipeex.utils import centr_dsd, DSD_bins
    
    #clean variable names and their mks units (and scale factors into those \
    #units), as well as column indices of relevant values in the raw files.
    key_ind_dict = {'MET':\
                        {'var_names':['time', 'sectime', 'temp', 'pres', \
                            'alt', 'vert_wind_vel', 'nconc_cdp', \
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
                        'var_scale':[1.] + [1.e-6 for i in range(9, 100)], \
                        'var_shift':[0. for i in range(92)]}}
    
    #get names of data files with no issues (see notes)
    with open('good_csv_files.txt','r') as readFile:
        good_csv_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_csv_filenames:
        #pick out relevant datasets and load raw .npy files
        basename = filename[0:-4] + '.npy'
        if 'dsd' in basename:
            setname='DSD'
        else:
            setname='MET'

        raw_dict = np.load(input_data_dir+basename, \
                allow_pickle=True).item()
        proc_dict = {'dataset':basename[:-4], \
                        'data':{}, \
                        'units':{}} 

        #select desired data from raw files and reformat slightly
        var_names = key_ind_dict[setname]['var_names']
        var_units = key_ind_dict[setname]['var_units']
        var_inds = key_ind_dict[setname]['var_inds']
        var_scale = key_ind_dict[setname]['var_scale']
        var_shift = key_ind_dict[setname]['var_shift']
        for i in range(len(var_names)):
            if setname == 'MET':
                proc_dict['data'][var_names[i]] = var_scale[i] \
                    *(raw_dict['data'][:,var_inds[i]] \
                    + var_shift[i])
            else:
                if i == 0: #time variable
                    proc_dict['data'][var_names[i]] = var_scale[i] \
                        *(raw_dict['data'][var_inds[i],:] \
                        + var_shift[i])
                else: #mass distribution vars; convert to nconc (var_shift=0)
                    proc_dict['data'][var_names[i]] = var_scale[i] \
                        *(raw_dict['data'][var_inds[i],:]/(4./3.*np.pi \
                        *centr_dsd[i-1]**3.))
            proc_dict['units'][var_names[i]] = var_units[i]
        
        #save processed files
        datestr = raw_dict['flight_date']
        np.save(output_data_dir+setname+'_'+datestr, proc_dict)

    #now calculate LWC values for DSD files 
    #centr_dsd = centr_dsd/2.
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
        
        #get met data for that date
        filename = DATA_DIR + 'npy_proc/MET_' + date + '.npy' 
        metdata = np.load(filename, allow_pickle=True).item()

        #get dsd data and create new file with lwc entry
        filename = DATA_DIR + 'npy_proc/' + setname + '_' \
                + date + '.npy'
        dataset = np.load(filename, allow_pickle=True).item()
        updated_dataset = dataset.copy()
            
        #calculate lwc...much less complicated than HALO because times
        #are synced and there's only one instrument for each part of the
        #water droplet size distribution. also no corrections to make
        #afaik right now.
        cloud_water_dens = np.zeros(metdata['data']['time'].shape)
        rain_water_dens = np.zeros(metdata['data']['time'].shape)
        rho_air = metdata['data']['pres']/(R_a*metdata['data']['temp'])

        for i in range(1, 92):
            if i < 31: #CDP range; cloud droplets
                var_key = 'nconc_' + str(i)
                cloud_water_dens += dataset['data'][var_key] \
                    *(4./3.*np.pi*(centr_dsd[i-1])**3.*rho_w)
            else: #CIP range; rain drops
                var_key = 'nconc_' + str(i)
                rain_water_dens += dataset['data'][var_key] \
                    *(4./3.*np.pi*(centr_dsd[i-1])**3.*rho_w)

        updated_dataset['data']['lwc_cloud'] = cloud_water_dens/rho_air
        updated_dataset['units']['lwc_cloud'] = 'g/g'
        updated_dataset['data']['lwc_rain'] = rain_water_dens/rho_air
        updated_dataset['units']['lwc_rain'] = 'g/g'
            
        np.save(filename, updated_dataset)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
