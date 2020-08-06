"""
First round of CAIPEEX data processing: .csv --> .npy, stripping unnecessary \
metadata from .csv files. 

Input location: /data/caipeex/ames
Output location: /data/caipeex/npy_raw
Output format: .npy file containing one dictionary formatted as: \
        {"flight_date": ['YYYY', 'MM', 'DD'], \
         "var_names": ['<full var name 1>', ...], \
         "data": <numpy array with columns labeled by var_names>}
"""
import csv
import numpy as np

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR

input_data_dir =  DATA_DIR + 'csv/'
output_data_dir = DATA_DIR + 'npy_raw/'

def main():
    """
    extract flight date, variable names, and data from csv files to numpy \
    files. also convert error codes to np.nan.
    """

    #get names of data files with no issues (see notes)
    with open('good_csv_files.txt','r') as readFile:
        good_csv_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_csv_filenames:
        basename = filename[0:-4] + '.npy' #replace '.csv' with '.npy'
        flight_date = filename[0:8] #yyyymmdd
        data_arr = []
        if 'dsd' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            data_arr = np.array(data_arr)
            var_names = np.concatenate((data_arr[0:9, 0], data_arr[9:, 1]))
            data = np.array(data_arr[:, 2:], dtype=float)
        else:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            data_arr = np.array(data_arr)
            var_names = data_arr[0, :]
            data = np.array(data_arr[1:, :], dtype=float)

        #save all fields in .npy format
        data_dict = {"flight_date":flight_date, "var_names":var_names, "data":data}
        np.save(output_data_dir+basename, data_dict)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
