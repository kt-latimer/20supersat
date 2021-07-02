"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_ss_qss

versionstr = 'v2_'

###
### physical constants
###
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_e = 6.3781e6 #radius of Earth (m)
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

lwc_cutoff = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

def main():
    
    for case_label in case_label_dict.keys():
        calc_avg_ssqss(case_label, case_label_dict[case_label])

def calc_avg_ssqss(case_label, case_dir_name):

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    filter_inds = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff), \
                            (temp > 273), \
                            (w > w_cutoff), \
                            (z > 1000), \
                            (z < 1500)))
    
    print(case_label)
    print(np.nanmean(ss_qss[filter_inds]))

if __name__ == "__main__":
    main()
