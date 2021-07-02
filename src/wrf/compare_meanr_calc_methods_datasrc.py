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
from wrf.dsd_data_functions import get_bin_nconc, get_bin_vent_coeff
from wrf.met_data_functions import get_dyn_visc
from wrf.ss_functions import get_lwc, get_meanr, get_nconc

versionstr = 'v3_'

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

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

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

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

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

    #meanr = get_meanr(dsdsum_vars, cutoff_bins, incl_rain, incl_vent) 
    #nconc = get_nconc(dsdsum_vars, cutoff_bins, incl_rain, incl_vent) 
    lwc = get_lwc(met_vars, dsdsum_vars, False, False, False)
    pres = met_vars['pres'][...]
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    z = met_vars['z'][...]

    #get values for calculating ventilation coefficient
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air**2./(eta**4.*g*rho_w) #pr&kl p 418

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
                            (z > 1500), \
                            (z < 2500)))

    #meanr = meanr[filter_inds]
    #nconc = nconc[filter_inds]
    
    #fnr_sum = np.zeros(np.shape(meanr))
    finiri_avg = np.zeros(33)
    finiri_avg_nanmean = np.zeros(33)

    for i in range(1, 34):
        nconc_i = get_bin_nconc(i, input_vars, rho_air)
        f_i = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                                    pres, rho_air, temp)
        nconc_i = nconc_i[filter_inds]
        f_i = f_i[filter_inds]
        #fnr_sum += bin_radii[i-1]*f_i*nconc_i
        finiri_avg[i-1] = np.mean(nconc_i*f_i)
        finiri_avg_nanmean[i-1] = np.nanmean(nconc_i*f_i)

    #ss_fn_meanr_dict = {'data': meanr*nconc}
    #fnr_sum_dict = {'data': fnr_sum}
    finiri_avg_dict = {'data': finiri_avg}
    finiri_avg_nanmean_dict = {'data': finiri_avg_nanmean}

    #close files for memory
    input_file.close()

    #ss_fn_meanr_filename = DATA_DIR + versionstr + 'ss_fn_meanr_' \
    #                                        + case_label + '_data.npy'
    #fnr_sum_filename = DATA_DIR + versionstr + 'fnr_sum_' \
    #                                + case_label + '_data.npy'
    finiri_avg_filename = DATA_DIR + versionstr + 'finiri_avg_' \
                                    + case_label + '_data.npy'
    finiri_avg_nanmean_filename = DATA_DIR + versionstr + 'finiri_avg_nanmean_' \
                                    + case_label + '_data.npy'

    #np.save(ss_fn_meanr_filename, ss_fn_meanr_dict)
    #np.save(fnr_sum_filename, fnr_sum_dict)
    np.save(finiri_avg_filename, finiri_avg_dict)
    np.save(finiri_avg_nanmean_filename, finiri_avg_nanmean_dict)

if __name__ == "__main__":
    main()
