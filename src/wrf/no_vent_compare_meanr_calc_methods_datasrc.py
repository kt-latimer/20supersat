"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.dsd_data_functions import get_bin_nconc
from wrf.ss_functions import get_lwc

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

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]
            
lwc_cutoff = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

cutoff_bins = False 
incl_rain = True 
incl_vent = True 
full_ss = True 

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2. 

bin_widths = bin_radii*(2.**(1./6.) - 2.**(-1./6.)) #edges are geometric means
                                                    #of center radius values
bin_widths_um = bin_widths*1.e6
log_bin_widths = np.array([np.log10(2.**(1./3.)) for i in range(33)])

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_meanr_dicts(case_label, \
                                case_label_dict[case_label])

def make_and_save_meanr_dicts(case_label, case_dir_name):

    print(case_label)

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars', 'r')
    dsdsum_vars = dsdsum_file.variables

    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    pres = met_vars['pres'][...]
    rho_air = met_vars['rho_air'][...]
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
                            (z > 3000), \
                            (z < 4000)))

    #meanr = meanr[filter_inds]
    #nconc = nconc[filter_inds]
    
    #fnr_sum = np.zeros(np.shape(meanr))
    niri_avg = np.zeros(33)
    niri_avg_nanmean = np.zeros(33)

    for i in range(1, 34):
        nconc_i = get_bin_nconc(i, input_vars, rho_air)

        nconc_i = nconc_i[filter_inds]
        niri_avg[i-1] = np.mean(nconc_i)
        niri_avg_nanmean[i-1] = np.nanmean(nconc_i)

    #ss_fn_meanr_dict = {'data': meanr*nconc}
    #fnr_sum_dict = {'data': fnr_sum}
    niri_avg_dict = {'data': niri_avg}
    niri_avg_nanmean_dict = {'data': niri_avg_nanmean}

    #close files for memory
    input_file.close()

    niri_avg_filename = DATA_DIR + versionstr + 'niri_avg_' \
                                    + case_label + '_data.npy'
    niri_avg_nanmean_filename = DATA_DIR + versionstr + 'niri_avg_nanmean_' \
                                    + case_label + '_data.npy'

    np.save(niri_avg_filename, niri_avg_dict)
    np.save(niri_avg_nanmean_filename, niri_avg_nanmean_dict)

if __name__ == "__main__":
    main()
