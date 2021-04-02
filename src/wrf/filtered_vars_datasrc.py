"""
make smaller data files for quantities after applying LWC, vertical wind
velocity, and temperature filters
"""
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

lwc_filter_val = 1.e-4
w_cutoff = 1
z_min = -100
z_max = 6500

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    filtered_data_dict = {'Polluted': None, 'Unpolluted': None}
    for case_label in case_label_dict.keys():
        filtered_data_dict[case_label] = get_filtered_data(case_label, \
                                case_label_dict[case_label], cutoff_bins, \
                                full_ss, incl_rain, incl_vent)

    filename = 'filtered_data_dict_v3.npy'
    np.save(DATA_DIR + filename, filtered_data_dict)

def get_filtered_data(case_label, case_dir_name, cutoff_bins, \
                                full_ss, incl_rain, incl_vent):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lh = met_vars['LH'][...]
    temp = met_vars['temp'][...]

    #filtering to determine total condensational latent heating
    lh_filter_inds = np.logical_and.reduce((
                    (lh > 0), \
                    (temp > 273)))
    lh_tot = np.sum(lh[lh_filter_inds])

    del lh_filter_inds #for memory

    #get relevant physical qtys, cont'd
    #lwc = met_vars['LWC_cloud'][...]+ met_vars['LWC_rain'][...]
    #lwc = met_vars['LWC_cloud'][...]
    #lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100
    pres = met_vars['pres'][...]
    w = met_vars['w'][...]
    x = met_vars['x'][...]
    y = met_vars['y'][...]
    x = np.transpose(np.tile(x, [66, 1, 1, 1]), [1, 0, 2, 3])
    y = np.transpose(np.tile(y, [66, 1, 1, 1]), [1, 0, 2, 3])
    z = met_vars['z'][...]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #get z bins before filtering
    z_bins = get_z_bins(z)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    lh = lh[filter_inds]
    lwc = lwc[filter_inds]
    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]
    temp = temp[filter_inds]
    w = w[filter_inds]
    x = x[filter_inds]
    y = y[filter_inds]
    z = z[filter_inds]

    case_filtered_data_dict = {'lh_tot': lh_tot, 'lh': lh, 'lwc': lwc, \
                    'ss_qss': ss_qss, 'ss_wrf': ss_wrf, 'temp': temp, \
                    'w': w, 'x': x, 'y': y, 'z': z, 'z_bins': z_bins, \
                    'pres': pres}

    return case_filtered_data_dict

def get_z_bins(z):

    n_bins = np.shape(z)[1]
    n_edges = n_bins + 1
    avg_z = np.array([np.mean(z[:, i, :, :]) for i in range(n_bins)])
    z_bins = [] 

    for i in range(1, n_bins):
        layer_geom_mean = np.sqrt(avg_z[i-1]*avg_z[i])
        if layer_geom_mean < z_max:
            z_bins.append(layer_geom_mean)
        else:
            break

    z_bins.insert(0, avg_z[0]*np.sqrt(avg_z[0]/avg_z[1]))

    return np.array(z_bins)

if __name__ == "__main__":
    main()
