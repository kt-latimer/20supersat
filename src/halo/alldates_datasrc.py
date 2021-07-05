"""
Consolidate data from all HALO flight dates used in our study into a single
file for each instrument.
"""
import numpy as np

from halo import DATA_DIR, FIG_DIR

output_data_dir = DATA_DIR + 'npy_proc/'

def main():
    
    tot_ADLR_dict, tot_CAS_dict, tot_CDP_dict, tot_CIP_dict = \
                                    get_dicts_from_all_dates()
    save_processed_file('ADLR', 'alldates', tot_ADLR_dict)
    save_processed_file('CAS', 'alldates', tot_CAS_dict)
    save_processed_file('CDP', 'alldates', tot_CDP_dict)
    save_processed_file('CIP', 'alldates', tot_CIP_dict)

def get_dicts_from_all_dates():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    tot_ADLR_dict, tot_CAS_dict, tot_CDP_dict, tot_CIP_dict = \
                            get_dicts_from_one_date(good_dates[0])

    for date in good_dates[1:]:
        ADLR_dict, CAS_dict, CDP_dict, CIP_dict = get_dicts_from_one_date(date)
        tot_ADLR_dict = update_tot_dict(tot_ADLR_dict, ADLR_dict)
        tot_CAS_dict = update_tot_dict(tot_CAS_dict, CAS_dict)
        tot_CDP_dict = update_tot_dict(tot_CDP_dict, CDP_dict)
        tot_CIP_dict = update_tot_dict(tot_CIP_dict, CIP_dict)

    return tot_ADLR_dict, tot_CAS_dict, tot_CDP_dict, tot_CIP_dict

def get_dicts_from_one_date(date):
    
    ADLR_file = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CDP_file = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    CDP_dict = np.load(CDP_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    return ADLR_dict, CAS_dict, CDP_dict, CIP_dict

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
