"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():
    
    filename = 'filtered_data_dict.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    ss_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        case_data_dict = data_dict[case_label]
        
        ss_qss = case_data_dict['ss_qss']
        ss_wrf = case_data_dict['ss_wrf']

        case_qss_filename = 'ss_qss_' + case_label + '_data'
        case_wrf_filename = 'ss_wrf_' + case_label + '_data'

        f_qss = open(DATA_DIR + case_qss_filename, 'bw')
        ss_qss.filled().tofile(f_qss)

        f_wrf = open(DATA_DIR + case_wrf_filename, 'bw')
        ss_wrf.filled().tofile(f_wrf)

if __name__ == "__main__":
    main()
