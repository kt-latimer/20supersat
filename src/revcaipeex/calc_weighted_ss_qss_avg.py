"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import numpy as np

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_ss_vs_t, get_lwc

lwc_filter_val = 1.e-4
w_cutoff = 2
z_max = 6500

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))

cutoff_bins = True
incl_rain = False 
incl_vent = False
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    pres_alldates = None 
    ss_qss_alldates = None
    temp_alldates = None
    w_alldates = None
    z_alldates = None

    for date in good_dates:
        pres, temp, ss_qss, w, z = get_one_day_data(date)
        pres_alldates = add_to_alldates_array(pres, pres_alldates)
        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)
        temp_alldates = add_to_alldates_array(temp, temp_alldates)
        w_alldates = add_to_alldates_array(w, w_alldates)
        z_alldates = add_to_alldates_array(z, z_alldates)

    calc_weighted_ss_qss_avg(pres_alldates, temp_alldates, \
                    ss_qss_alldates, w_alldates, z_alldates)

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def get_one_day_data(date):

    metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
    met_dict = np.load(metfile, allow_pickle=True).item()
    cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cpd_dict = np.load(cpdfile, allow_pickle=True).item()

    lwc = get_lwc(cpd_dict,cutoff_bins)
    pres = met_dict['data']['pres']
    temp = met_dict['data']['temp']
    w = met_dict['data']['w']
    z = met_dict['data']['alt']
    ss_qss = get_ss_vs_t(met_dict, cpd_dict, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #there's a weird outlier which the third line removes
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (ss_qss < 100), \
                    (temp > 273)))

    pres = pres[filter_inds]
    ss_qss = ss_qss[filter_inds]
    temp = temp[filter_inds]
    w = w[filter_inds]
    z = z[filter_inds]

    return pres, temp, ss_qss, w, z

def calc_weighted_ss_qss_avg(pres, temp, ss_qss, w, z):

    print(np.mean(temp), np.median(temp), np.min(temp), np.max(temp))

    z_bins = get_z_bins(z)

    qvstar = get_qvstar(pres, temp)
    dt = np.ones(np.shape(ss_qss))
    dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                    range(np.shape(z_bins)[0] - 1)])
    ss_qss_wtd_avg = np.sum(dt\
                            *qvstar\
                            *ss_qss)\
                        /np.sum(dt\
                            *qvstar)
    print('allpts')
    print(ss_qss_wtd_avg)

    up10perc_cutoff = np.percentile(w, 90)
    up10perc_inds = w > up10perc_cutoff

    pres = pres[up10perc_inds]
    ss_qss = ss_qss[up10perc_inds]
    temp = temp[up10perc_inds]
    print(np.mean(temp), np.median(temp), np.min(temp), np.max(temp))
    qvstar = get_qvstar(pres, temp)
    dt = np.ones(np.shape(ss_qss))
    ss_qss_wtd_avg = np.sum(dt\
                            *qvstar\
                            *ss_qss)\
                        /np.sum(dt\
                            *qvstar)
    print('up10perc')
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

def get_z_bins(z):
    
    h_z, z_bins = np.histogram(z, bins=30, density=True)

    return z_bins

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

if __name__ == "__main__":
    main()
