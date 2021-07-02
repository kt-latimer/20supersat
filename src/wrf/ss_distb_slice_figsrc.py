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
                            
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

cutoff_bins = True 
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
        dsd_dict, vent_dsd_dict = get_dsd_dicts(case_label, \
                                case_label_dict[case_label])
        make_and_save_avg_dsd(case_label, dsd_dict, vent_dsd_dict)

def get_dsd_dicts(case_label, case_dir_name):

    print(case_label)

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    pres = met_vars['pres'][...]
    rho_air = met_vars['rho_air'][...]
    ss_wrf = met_vars['ss_wrf'][...]
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

    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    filter_inds = np.logical_and.reduce(( \
                            (temp > 273), \
                            (w > w_cutoff), \
                            (z > 3000), \
                            (z < 4000)))

    dsd_dict = {'mean': [], 'std': [], 'median': [], \
                        'up_quart': [], 'lo_quart': []}
    vent_dsd_dict = {'mean': [], 'std': [], 'median': [], \
                            'up_quart': [], 'lo_quart': []}

    for i in range(1, 34):
        nconc_i = get_bin_nconc(i, input_vars, rho_air)
        f_i = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                                    pres, rho_air, temp)
        dsd_dict['mean'].append(np.mean(nconc_i[filter_inds]))
        dsd_dict['std'].append(np.std(nconc_i[filter_inds]))
        dsd_dict['median'].append(np.median(nconc_i[filter_inds]))
        dsd_dict['lo_quart'].append(np.median(nconc_i[filter_inds]) - \
                                    np.percentile(nconc_i[filter_inds], 25))
        dsd_dict['up_quart'].append(np.percentile(nconc_i[filter_inds], 75) - \
                                    np.median(nconc_i[filter_inds]))

        vent_dsd_dict['mean'].append(np.mean(f_i[filter_inds]*nconc_i[filter_inds]))
        vent_dsd_dict['std'].append(np.std(f_i[filter_inds]*nconc_i[filter_inds]))
        vent_dsd_dict['median'].append(np.median(f_i[filter_inds]*nconc_i[filter_inds]))
        vent_dsd_dict['lo_quart'].append(np.median(f_i[filter_inds]*nconc_i[filter_inds]) - \
                                np.percentile(f_i[filter_inds]*nconc_i[filter_inds], 25))
        vent_dsd_dict['up_quart'].append(np.percentile(f_i[filter_inds]*nconc_i[filter_inds], 75) - \
                                np.median(f_i[filter_inds]*nconc_i[filter_inds]))

    for key in dsd_dict.keys():
        dsd_dict[key] = np.array(dsd_dict[key])

    for key in vent_dsd_dict.keys():
        vent_dsd_dict[key] = np.array(vent_dsd_dict[key])

    #close files for memory
    input_file.close()

    case_dsd_filename = DATA_DIR + versionstr + 'dsd_dict_slice_' \
                                            + case_label + '_data'
    case_vent_dsd_filename = DATA_DIR + versionstr + 'vent_dsd_dict_slice_' \
                                                        + case_label + '_data'

    np.save(case_dsd_filename, dsd_dict)
    np.save(case_vent_dsd_filename, vent_dsd_dict)

    return dsd_dict, vent_dsd_dict

def make_and_save_avg_dsd(case_label, dsd_dict, vent_dsd_dict):

    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)

    ax11.errorbar(bin_radii*1.e6, dsd_dict['mean']*1.e-6, \
                        yerr = dsd_dict['std']*1.e-6, fmt='o')
    ax12.errorbar(bin_radii*1.e6, dsd_dict['mean']*bin_radii, \
                        yerr = dsd_dict['std']*bin_radii, fmt='o')
    ax21.errorbar(bin_radii*1.e6, dsd_dict['median']*1.e-6, \
                        yerr = [dsd_dict['lo_quart']*1.e-6, \
                        dsd_dict['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(bin_radii*1.e6, dsd_dict['median']*bin_radii, \
                        yerr = [dsd_dict['lo_quart']*bin_radii, \
                        dsd_dict['up_quart']*bin_radii], fmt='o')

    outfile = FIG_DIR + versionstr + 'avg_dsd_slice_' + \
                                case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)

    ax11.errorbar(bin_radii*1.e6, vent_dsd_dict['mean']*1.e-6, \
                        yerr = vent_dsd_dict['std'], fmt='o')
    ax12.errorbar(bin_radii*1.e6, vent_dsd_dict['mean']*bin_radii, \
                        yerr = vent_dsd_dict['std']*bin_radii, fmt='o')
    ax21.errorbar(bin_radii*1.e6, vent_dsd_dict['median']*1.e-6, \
                        yerr = [vent_dsd_dict['lo_quart']*1.e-6, \
                        vent_dsd_dict['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(bin_radii*1.e6, vent_dsd_dict['median']*bin_radii, \
                        yerr = [vent_dsd_dict['lo_quart']*bin_radii, \
                        vent_dsd_dict['up_quart']*bin_radii], fmt='o')

    outfile = FIG_DIR + versionstr + 'avg_vent_dsd_slice_' + \
                                    case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

if __name__ == "__main__":
    main()
