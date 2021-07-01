"""
Consolidate data from all HALO flight dates used in our study into a single
file for each instrument.
"""
import numpy as np

from halo import DATA_DIR, FIG_DIR

output_data_dir = DATA_DIR + 'npy_proc/'

def main():
    
    tot_adlr_dict, tot_cas_dict, tot_cdp_dict, tot_cip_dict = \
                                    get_dicts_from_all_dates()
    save_processed_file('ADLR', 'alldates', tot_adlr_dict)
    save_processed_file('CAS', 'alldates', tot_cas_dict)
    save_processed_file('CDP', 'alldates', tot_cdp_dict)
    save_processed_file('CIP', 'alldates', tot_cip_dict)

def get_dicts_from_all_dates():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    tot_adlr_dict, tot_cas_dict, tot_cdp_dict, tot_cip_dict = \
                            get_dicts_from_one_date(good_dates[0])

    for date in good_dates[1:]:
        adlr_dict, cas_dict, cdp_dict, cip_dict = get_dicts_from_one_date(date)
        tot_adlr_dict = update_tot_dict(tot_adlr_dict, adlr_dict)
        tot_cas_dict = update_tot_dict(tot_cas_dict, cas_dict)
        tot_cdp_dict = update_tot_dict(tot_cdp_dict, cdp_dict)
        tot_cip_dict = update_tot_dict(tot_cip_dict, cip_dict)

    return tot_adlr_dict, tot_cas_dict, tot_cdp_dict, tot_cip_dict

def get_dicts_from_one_date(date):
    
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdp_dict = np.load(cdpfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    return adlr_dict, cas_dict, cdp_dict, cip_dict

def update_tot_dict(tot_data_dict, data_dict):

    keys = tot_data_dict['data'].keys()

    for key in keys:
        tot_data_dict['data'][key] = np.concatenate( \
            (tot_data_dict['data'][key], data_dict['data'][key]))

    return tot_data_dict

def save_processed_file(setname, date, processed_data_dict):

    processed_filename = setname + '_' + date + '.npy'
    np.save(output_data_dir + processed_filename, processed_data_dict)
                            
if __name__ == "__main__":
    main()
