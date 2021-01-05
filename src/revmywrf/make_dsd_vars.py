"""
make dsd data files
"""
from netCDF4 import Dataset, MFDataset
import numpy as np

from revmywrf import BASE_DIR, DATA_DIR, FIG_DIR 
from revmywrf.dsd_data_functions import get_bin_nconc, get_bin_vent_coeff

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

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

dim_4d = ('Time', 'bottom_top', 'south_north', 'west_east')

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

def main():

    for case_label in case_label_dict.keys():
        make_dsd_file(case_label, case_label_dict[case_label])

def make_dsd_file(case_label, case_dir_name):

    #get input file variables (naming a bit confusing here but trying
    #to maintain consistency with make_met_vars code...)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables
    met_input_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_input_vars = met_input_file.variables

    #get air density
    pres_data = met_input_vars['pres'][...]
    rho_air_data = met_input_vars['rho_air'][...]
    temp_data = met_input_vars['temp'][...]

    #close met input file for memory
    met_input_file.close()

    #get values for calculating ventilation coefficient
    eta = get_dyn_visc(temp_data)
    sigma = sum([sigma_coeffs[i]*(temp_data - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air_data*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air_data**2./(eta**4.*g*rho_w) #pr&kl p 418
    
    #make output file
    output_file = Dataset(DATA_DIR + case_dir_name \
                    + 'wrfout_d01_dsd_vars', 'w')

    #make file dimensions
    output_file.createDimension('west_east', 450)
    output_file.createDimension('south_north', 450)
    output_file.createDimension('bottom_top', 66)
    output_file.createDimension('Time', 84)

    for i in range(1, 33):
        nconc_var_name = 'nconc_' + str(i)
        f_var_name = 'f_' + str(i)
        nconc_var = output_file.createVariable(nconc_var_name, \
                                    np.dtype('float32'), dim_4d) 
        f_var = output_file.createVariable(f_var_name, \
                                    np.dtype('float32'), dim_4d) 
        nconc_var_data = get_bin_nconc(i, input_vars, rho_air_data)
        f_var_data = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                        pres_data, rho_air_data, temp_data)
        nconc_var[:] = nconc_var_data
        f_var[:] = f_var_data

    input_file.close()
    output_file.close()

def get_dyn_visc(temp):
    """
    get dynamic viscocity as a function of temperature (from pruppacher and
    klett p 417)
    """
    eta = np.piecewise(temp, [temp < 273, temp >= 273], \
                        [lambda temp: (1.718 + 0.0049*(temp - 273) \
                                    - 1.2e-5*(temp - 273)**2.)*1.e-5, \
                        lambda temp: (1.718 + 0.0049*(temp - 273))*1.e-5])
    return eta

if __name__ == "__main__":
    main()
