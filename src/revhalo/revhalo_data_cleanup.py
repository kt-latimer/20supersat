"""
First round of HALO data processing: .ames --> .npy, stripping unnecessary \
metadata from .ames files. 

Input location: /data/halo/ames (just using original raw files to save space)
Output location: /data/revhalo/npy_raw
Output format: .npy file containing one dictionary formatted as: \
        {"date": ['YYYY', 'MM', 'DD'], \
         "var_names": ['<full var name 1>', ...], \
         "data": <numpy array with columns labeled by var_names>}

TODO: rewrite this code in self-documented style (relatively simple proceedure
and works ok so low priority for now)
"""
import numpy as np

from revhalo import BASE_DIR, DATA_DIR, FIG_DIR

#using halo copy of ames files for now
input_data_dir =  '/home/klatimer/proj/20supersat/data/halo/' + 'ames/'
output_data_dir = DATA_DIR + 'npy_raw/'

def main():
    """
    extract flight date, variable names, and data from ames files to numpy \
    files. also convert error codes to np.nan.
    """

    #get names of data files with no issues (see notes)
    with open('good_ames_filenames.txt','r') as readFile:
        good_ames_filenames = [line.strip() for line in readFile.readlines()]
    readFile.close()

    #create .npy file for each .ames file in good_ames_filenames
    for filename in good_ames_filenames:
        
        basename = filename[0:len(filename)-5]
        
        with open(input_data_dir+filename, 'r') as readFile:
            print(filename)
            lines = readFile.readlines()
            num_header_lines = int(lines[0].split()[0])

            #get flight date
            flight_date = lines[6].split()[0:3]
            if len(flight_date[1]) == 1:
                flight_date[1] = '0' + flight_date[1]
            if len(flight_date[2]) == 1:
                flight_date[2] = '0' + flight_date[2]
            
            #get error code values and check if they are all the same
            n_vars = int(lines[9]) + 1 #counting independent variable 
            first_scale_factor_line = lines[10].split()
            condensed_notation = (len(first_scale_factor_line) != 1)
            if condensed_notation: #ames 'condensed' notation
                err_vals = np.array(lines[11].split()).astype(np.float)
            else:
                err_vals = np.array([float(lines[9 + n_vars + i]) for i in \
                        range(n_vars - 1)])
            err_val = err_vals[0]
            for i, val in enumerate(err_vals):
                if val != err_val:
                    print('WARNING: not all error values equal for', basename)

            #get full variable names
            if condensed_notation:
                var_names = [lines[8].strip()]+[lines[12 + i].strip() for i in range(n_vars - 1)]
            else:
                var_names = [lines[8].strip()]+[lines[10 + 2*n_vars + i].strip() for i in range(n_vars - 1)]

            #get numerical data
            data = np.array([np.array(line.split()).astype(np.float) for line \
                    in lines[num_header_lines:-1] if line.split() != []])

            #replace all error codes with uniform value of np.nan
            data = np.where(data==err_val, np.nan, data)

            #save all fields in .npy format
            data_dict = {"date":flight_date, "var_names":var_names, "data":data}
            np.save(output_data_dir+basename, data_dict)
        readFile.close()

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
