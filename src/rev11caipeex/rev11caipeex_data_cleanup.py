"""
CAIPEEX data processing for 2011 data: .csv --> .npy, stripping unnecessary \
metadata from .csv files. 

Input location: /data/rev11caipeex/ames
Output location: /data/revrev11caipeex/npy_raw
Output format: .npy file containing one dictionary formatted as: \
        {"date": ['YYYY', 'MM', 'DD'], \
         "var_names": ['<full var name 1>', ...], \
         "data": <numpy array with columns labeled by var_names>}
"""
import csv
import numpy as np

from rev11caipeex import BASE_DIR, DATA_DIR, FIG_DIR

input_data_dir =  DATA_DIR + 'csv/'
output_data_dir = DATA_DIR + 'npy_raw/'

def main():
    """
    extract flight date, variable names, and data from csv files to numpy \
    files. also convert error codes to np.nan.
    """

    #get names of data files with no issues (see notes)
    with open('good_csv_filenames.txt','r') as readFile:
        good_csv_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_csv_filenames:
        basename = filename[0:-4] + '.npy' #replace '.csv' with '.npy'
        if '20111027' in filename or 'RF40' in filename:
            flight_date = '20111027' #yyyymmdd
        if '20111024' in filename or 'RF36' in filename:
            flight_date = '20111024' #yyyymmdd
        data_arr = []
        if 'ALL' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            data_arr = np.array(data_arr)
            var_names = data_arr[0, 0:9]
            data = np.array(data_arr[1:, :], dtype=float)
        elif 'FSSP' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            #idk why this is required but it is
            data_arr[2] = data_arr[2][:-1]
            data_arr = np.array(data_arr[2:])
            var_names = data_arr[0, 0:32]
            data = np.array(data_arr[1:, :], dtype=float)
        elif 'CIP' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            #idk why this is required but it is
            data_arr[2] = data_arr[2][:-1]
            #print(data_arr)
            data_arr = np.array(data_arr[2:])
            var_names = data_arr[0, 0:37]
            data = np.array(data_arr[1:, :], dtype=float)
        elif 'DMA' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                lines = readFile.readlines()
            data_arr = np.array([np.concatenate(( \
                    np.array([convert_time_str_to_float(line.split()[0])]), \
                    np.array(line.split()[1:]).astype(np.float))) for line \
                    in lines[1:] if line.split() != []])
            #idk why this is required but it is
            #data_arr[2] = data_arr[2][:-1]
            #print(data_arr)
            #data_arr = np.array(data_arr[2:])
            var_names = data_arr[0]
            data = np.array(data_arr[1:, :])
        elif 'PCASP' in filename:
            with open(input_data_dir+filename, 'r') as readFile:
                csvreader = csv.reader(readFile, delimiter=',')
                for row in csvreader:
                    data_arr.append(row)
            if flight_date == '20111024':
                data_arr = np.array(data_arr[31:])
                var_names = data_arr[0][0:2]
                data = np.array(data_arr[1:, :], dtype=float)
            if flight_date == '20111027':
                data_arr = np.array(data_arr[20:])
                var_names = data_arr[0][0:2]
                data = np.array(data_arr[1:, :], dtype=float)

        #save all fields in .npy format
        data_dict = {"date":flight_date, "var_names":var_names, "data":data}
        np.save(output_data_dir+basename, data_dict)

def convert_time_str_to_float(time_str):

    hh = time_str[0:2]
    mm = time_str[3:5]
    ss = time_str[6:8]
    
    time_float = 3600.*float(hh) + 60*float(mm) + float(ss)

    return time_float

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
