"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
from halo.ss_functions import get_nconc_contribution_from_cas_var, \
                                get_meanr_contribution_from_cas_var, \
                                get_nconc_contribution_from_cip_var, \
                                get_meanr_contribution_from_cip_var, \
                                get_spliced_cas_and_cip_dicts
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[1]
magma_pink = colors_arr[5]
magma_orange = colors_arr[9]

change_cas_corr = True
cutoff_bins = False 
incl_rain = True 
incl_vent = False 

#physical constants
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of water (kg/m^3) 

CAS_bin_radii = np.sqrt((CAS_bins['upper']*CAS_bins['lower'])/4.)
CIP_bin_radii = np.sqrt((CIP_bins['upper']*CIP_bins['lower'])/4.)

rmax_vals = np.concatenate((CAS_bin_radii, CIP_bin_radii[1:])) 
lwc_cutoff_vals = [-1, 1.e-5, 1.e-4]

def main():

    cas_nconc_dict, cip_nconc_dict = get_nconc_dicts()
    casfile = DATA_DIR + 'alldates_cas_nconc.npy'
    cipfile = DATA_DIR + 'alldates_cip_nconc.npy'

    np.save(casfile, cas_nconc_dict)
    np.save(cipfile, cip_nconc_dict)

def get_nconc_dicts():

    tot_adlr_dict, tot_cas_dict, tot_cip_dict = get_dicts_from_all_dates()

    cas_nconc_dict = {}
    cip_nconc_dict = {}

    for i in range(5, 17):
        var_name = 'nconc_' + str(i)
        if change_cas_corr:
            var_name += '_corr'
        nconc = get_nconc_contribution_from_cas_var(var_name, \
                tot_adlr_dict, tot_cas_dict, change_cas_corr, \
                cutoff_bins, incl_rain, incl_vent)
        cas_nconc_dict[var_name] = nconc

    for i in range(1, 20):
        var_name = 'nconc_' + str(i)
        nconc = get_nconc_contribution_from_cip_var(var_name, \
                                    tot_adlr_dict, tot_cip_dict)
        cip_nconc_dict[var_name] = nconc

    return cas_nconc_dict, cip_nconc_dict 
             
def get_dicts_from_all_dates():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    tot_adlr_dict, tot_cas_dict, tot_cip_dict = \
            get_dicts_from_one_date(good_dates[0])

    for date in good_dates[1:]:
        adlr_dict, cas_dict, cip_dict = get_dicts_from_one_date(date)
        tot_adlr_dict = update_tot_dict(tot_adlr_dict, adlr_dict)
        tot_cas_dict = update_tot_dict(tot_cas_dict, cas_dict)
        tot_cip_dict = update_tot_dict(tot_cip_dict, cip_dict)

    return tot_adlr_dict, tot_cas_dict, tot_cip_dict

def get_dicts_from_one_date(date):
    
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 273), \
                            (w > 1)))
    
    adlr_dict = get_filtered_data(adlr_dict, filter_inds) 
    cas_dict = get_filtered_data(cas_dict, filter_inds) 
    cip_dict = get_filtered_data(cip_dict, filter_inds) 

    return adlr_dict, cas_dict, cip_dict

def update_tot_dict(tot_data_dict, data_dict):

    keys = tot_data_dict['data'].keys()

    for key in keys:
        tot_data_dict['data'][key] = np.concatenate( \
            (tot_data_dict['data'][key], data_dict['data'][key]))

    return tot_data_dict

def get_filtered_data(data_dict, filter_inds):

    keys = data_dict['data'].keys()

    for key in keys:
        data_dict['data'][key] = data_dict['data'][key][filter_inds]

    return data_dict

if __name__ == "__main__":
    main()
