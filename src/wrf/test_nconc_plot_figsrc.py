"""
calculating quantities from mid-radius sector WRF DSDs
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR 
from wrf.dsd_data_functions import get_bin_nconc

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

###
### bin sizes and regime params
###
bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

def main():

    for case_label in case_label_dict.keys():
        make_test_nconc_plot(case_label, case_label_dict[case_label])

def make_test_nconc_plot(case_label, case_dir_name):

    print(case_label)

    #get met input file variables (naming a bit confusing here but trying
    #to maintain consistency with make_net_vars code...)
    met_input_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_input_vars = met_input_file.variables

    #get relevant environmental data 
    rho_air_data = met_input_vars['rho_air'][...]
    temp = met_input_vars['temp'][...]
    w = met_input_vars['w'][...]

    filter_inds = np.logical_and(w > 1, temp > 273)

    #close met input file for memory
    met_input_file.close()
    
    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    test_nconc = get_bin_nconc(7, input_vars, rho_air_data)
    test_nconc = test_nconc[filter_inds]

    input_file.close()

    fig, ax = plt.subplots()

    ax.hist(test_nconc, bins=30, density=True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    outfile = FIG_DIR + 'test_nconc_plot_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
