"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
from netCDF4 import Dataset
import numpy as np

from revmywrf import DATA_DIR, FIG_DIR
from revmywrf.ss_qss_calculations import get_lwc, get_ss, linregress

lwc_filter_val = 1.e-4
w_cutoff = 2
z_max = 6500

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = True
incl_rain = True
incl_vent = True
full_ss = True

def main():
    
    for case_label in case_label_dict.keys():
        calc_weighted_ss_qss_avg(case_label)

def calc_weighted_ss_qss_avg(case_label):

    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = get_lwc(met_vars, dsdsum_vars, cutoff_bins, incl_rain, incl_vent)
    pres = met_vars['pres'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]
    ss_qss = get_ss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #before filtering, get z bins
    z_bins = get_z_bins(z)

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    del lwc #for memory

    pres[~filter_inds] = np.nan
    ss_qss[~filter_inds] = np.nan
    temp[~filter_inds] = np.nan
    ss_qss_tz = np.nanmean(ss_qss, axis=(2,3))
    A_tz = np.sum(filter_inds, axis=(2,3))
    pres_tz = np.nanmean(pres, axis=(2,3))
    temp_tz = np.nanmean(temp, axis=(2,3))
    qvstar_tz = get_qvstar(pres_tz, temp_tz)
    dt = np.ones(np.shape(ss_qss_tz))
    dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                    range(np.shape(z_bins)[0] - 1)])
    dz = np.tile(dz, [84, 1])
    z_max_ind = np.shape(dz)[1]
    print('ss', ss_qss_tz.shape)
    print('ss cutoff', ss_qss_tz[:, :z_max_ind].shape)
    print('dz', dz.shape)
    print('A_tz', A_tz.shape)
    print('qvstar_tz', qvstar_tz.shape)
    print('dt', dt.shape)
    ss_qss_wtd_avg = np.sum(dt[:, :z_max_ind]\
                            *dz\
                            #*A_tz[:, :z_max_ind]\
                            *qvstar_tz[:, :z_max_ind]\
                            *ss_qss_tz[:, :z_max_ind])\
                        /np.sum(dt[:, :z_max_ind]\
                            *dz\
                            #*A_tz[:, :z_max_ind]\
                            *qvstar_tz[:, :z_max_ind])
    print(case_label, 'allpts')
    print(ss_qss_wtd_avg)

    w_filt = w[filter_inds]
    up10perc_cutoff = np.percentile(w_filt, 90)
    up10perc_inds = np.logical_and.reduce((
                            (filter_inds), \
                            (w > up10perc_cutoff)))

    del w, w_filt #for memory

    pres[~up10perc_inds] = np.nan
    ss_qss[~up10perc_inds] = np.nan
    temp[~up10perc_inds] = np.nan
    ss_qss_tz = np.nanmean(ss_qss, axis=(2,3))
    A_tz = np.sum(up10perc_inds, axis=(2,3))
    pres_tz = np.nanmean(pres, axis=(2,3))
    temp_tz = np.nanmean(temp, axis=(2,3))
    qvstar_tz = get_qvstar(pres_tz, temp_tz)
    ss_qss_wtd_avg = np.sum(dt[:, :z_max_ind]\
                            *dz\
                            #*A_tz[:, :z_max_ind]\
                            *qvstar_tz[:, :z_max_ind]\
                            *ss_qss_tz[:, :z_max_ind])\
                        /np.sum(dt[:, :z_max_ind]\
                            *dz\
                            #*A_tz[:, :z_max_ind]\
                            *qvstar_tz[:, :z_max_ind])
    print(case_label, 'up10perc')
    print(ss_qss_wtd_avg)

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

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

if __name__ == "__main__":
    main()
