"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import numpy as np

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_ss_vs_t, get_lwc
import sys

#for plotting
versionstr = 'v1_'

lwc_filter_val = 1.e-4
w_cutoff = 2

z_min = -100
z_max = 6500

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
g = 9.8 #grav accel (m/s^2)
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))

def main():
    
    if len(sys.argv) > 1:
        versionnum = int(sys.argv[1])
        (dummy_bool, cutoff_bins, full_ss, \
            incl_rain, incl_vent) = get_boolean_params(versionnum)
        versionstr = 'v' + str(versionnum) + '_'

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    pres_alldates = None 
    ss_qss_alldates = None
    temp_alldates = None
    w_alldates = None
    z_alldates = None

    for date in good_dates:
        pres, temp, ss_qss, w, z = get_one_day_data(date, cutoff_bins, \
                                            full_ss, incl_rain, incl_vent)
        pres_alldates = add_to_alldates_array(pres, pres_alldates)
        ss_qss_alldates = add_to_alldates_array(ss_qss, ss_qss_alldates)
        temp_alldates = add_to_alldates_array(temp, temp_alldates)
        w_alldates = add_to_alldates_array(w, w_alldates)
        z_alldates = add_to_alldates_array(z, z_alldates)

    h_z, z_bins = np.histogram(z_alldates, bins=30, density=True)

    save_dT_profile_data(pres_alldates, temp_alldates, ss_qss_alldates, \
                            w_alldates, z_alldates, z_bins, versionstr)

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def get_one_day_data(date, cutoff_bins, full_ss, incl_rain, incl_vent):

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

def save_dT_profile_data(pres, temp, ss_qss, w, z, z_bins, versionstr):

    dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                    range(np.shape(z_bins)[0] - 1)])
    qvstar = get_qvstar(pres, temp)

    dT = get_dT(qvstar, ss_qss, temp)
    avg_dT, avg_temp, avg_z = get_avg_dT_and_temp_and_z(dT, temp, z, z_bins)
    notnan_inds = np.logical_not(np.isnan(avg_dT))
    avg_dT = avg_dT[notnan_inds]
    avg_temp = avg_temp[notnan_inds]
    avg_z = avg_z[notnan_inds]
    z_bins = get_adjusted_z_bins(notnan_inds, z_bins)
    dT_profile_data_dict = {'dT': avg_dT, 'temp': avg_temp, 'z': avg_z, 'z_bins': z_bins}
    print(dT_profile_data_dict)

    filename = versionstr + 'dT_profile_data_alldates.npy'
    np.save(DATA_DIR + filename, dT_profile_data_dict)

def get_qvstar(pres, temp):
    
    e_sat = get_e_sat(temp)

    return e_sat/pres*R_a/R_v
    
def get_e_sat(temp):

    e_sat = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))

    return e_sat

def get_up10perc_inds(w):

    w_cutoff = np.percentile(w, 90)
    up10perc_inds = w > w_cutoff

    return up10perc_inds 

def get_dT(qvstar, ss_qss, temp):

    dRH = ss_qss/100. #assuming parcel has RH=1 (as a fraction not percent)
    dT = qvstar*L_v/(C_ap + qvstar*L_v**2./(R_v*temp**2.))*dRH

    return dT

def get_avg_dT_and_temp_and_z(dT, temp, z, z_bins):

    n_bins = np.shape(z_bins)[0] - 1
    avg_dT = np.zeros(n_bins)
    avg_temp = np.zeros(n_bins)
    avg_z = np.zeros(n_bins)

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        if i == n_bins-1: #last upper bin edge is inclusive
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z <= upper_bin_edge)))
        else: 
            bin_filter = np.logical_and.reduce((
                            (z >= lower_bin_edge), \
                            (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_dT[i] = np.nan
            avg_temp[i] = np.nan
            avg_z[i] = np.nan
        else:
            dT_slice = dT[bin_filter]
            temp_slice = temp[bin_filter]
            z_slice = z[bin_filter]
            avg_dT[i] = np.nanmean(dT_slice)
            avg_temp[i] = np.nanmean(temp_slice)
            avg_z[i] = np.nanmean(z_slice)

    return avg_dT, avg_temp, avg_z

def get_adjusted_z_bins(notnan_inds, z_bins):

    z_bin_tuples = [(z_bins[i], z_bins[i+1]) \
                        for i in range(np.shape(z_bins)[0] - 1)]
    new_z_bin_tuples = []
    n_bins = len(z_bin_tuples)
    i = 0

    while i < n_bins:
        if notnan_inds[i]:
            new_z_bin_tuples.append(z_bin_tuples[i])
            i += 1
        else:
            lower_bin = z_bin_tuples[i-1]
            while not notnan_inds[i]:
                i += 1
            upper_bin = z_bin_tuples[i]
            midpoint = 0.5*(lower_bin[1] + upper_bin[0])
            new_z_bin_tuples[-1] = (lower_bin[0], midpoint)
            new_z_bin_tuples.append((midpoint, upper_bin[1]))
            i += 1
    
    new_z_bins = np.array([new_z_bin_tup[0] for new_z_bin_tup in \
                new_z_bin_tuples] + [new_z_bin_tuples[-1][1]])
    print(np.shape(new_z_bins))
    print(np.sum(notnan_inds))

    return new_z_bins

def get_boolean_params(versionnum):

    versionnum = versionnum - 1 #for modular arithmetic
    
    if versionnum > 23:
        return versionnum

    if versionnum < 12:
        change_cas_corr = False
    else:
        change_cas_corr = True

    if versionnum % 12 < 6:
        cutoff_bins = False
    else:
        cutoff_bins = True

    if versionnum % 6 < 3:
        full_ss = False
    else:
        full_ss = True

    if versionnum % 3 == 0:
        incl_rain = False
    else:
        incl_rain = True

    if versionnum % 3 == 2:
        incl_vent = True
    else:
        incl_vent = False

    return (change_cas_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

if __name__ == "__main__":
    main()
