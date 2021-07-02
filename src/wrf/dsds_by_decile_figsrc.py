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
from wrf.ss_functions import get_lwc, get_ss_qss

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

n_bins = 33
n_spectra = 20

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(n_bins)]) #bin diams in m
bin_radii = bin_diams/2. 
dlogDp = np.array([np.log10(2.**(1./3.)) for i in range(n_bins)])

##
## various series expansion coeffs - inline comment = page in pruppacher and klett
##
sigma_coeffs = [75.93, 0.115, 6.818e-2, 6.511e-3, \
                2.933e-4, 6.283e-6, 5.285e-8] #130

def main():
    
    for case_label in ['Unpolluted']:# case_label_dict.keys():
        spectrum_dict = make_and_save_spectrum_dict(case_label, \
                                case_label_dict[case_label])
        make_and_save_dsds_by_decile_graphs(case_label, spectrum_dict)

def make_and_save_spectrum_dict(case_label, case_dir_name):

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
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]

    #spacetime indices
    t_ind_vals = np.arange(84)
    x_ind_vals = np.arange(450)
    y_ind_vals = np.arange(450)
    z_ind_vals = np.arange(66)

    t, z, y, x = np.meshgrid(t_ind_vals, z_ind_vals, \
                y_ind_vals, x_ind_vals, indexing='ij')

    #get values for calculating ventilation coefficient
    eta = get_dyn_visc(temp)
    sigma = sum([sigma_coeffs[i]*(temp - 273)**i for i in \
                range(len(sigma_coeffs))])*1.e-3
    N_Be_div_r3 = 32*rho_w*rho_air*g/(3*eta**2.) #pr&kl p 417
    N_Bo_div_r2 = g*rho_w/sigma #pr&kl p 418
    N_P = sigma**3.*rho_air**2./(eta**4.*g*rho_w) #pr&kl p 418

    #for memory
    del t_ind_vals, x_ind_vals, y_ind_vals, z_ind_vals

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    env_filter = np.logical_and.reduce(( \
                            (lwc > lwc_cutoff), \
                            (temp > 273), \
                            (w > w_cutoff)))

    ss_qss = ss_qss[env_filter] 
    ss_wrf = ss_wrf[env_filter]
    w = w[env_filter] 
    t = t[env_filter] 
    x = x[env_filter] 
    y = y[env_filter] 
    z = z[env_filter] 

    decile_filters, rand_inds = get_decile_filters(ss_wrf)

    spectrum_dict = {'t_inds': [], 'x_inds': [], 'y_inds': [], \
                'z_inds': [], 'w_vals': [], 'ss_wrf_vals': [], \
                'ss_qss_vals': [], 'rand_inds': rand_inds}
    for i in range(1, 11):
        decile_key = 'decile_' + str(i)
        spectrum_dict[decile_key] = {'nconc': [[] for i in range(n_bins)], \
                                    'vent_coeff': [[] for i in range(n_bins)]}
    spectrum_dict = add_non_dsd_data(spectrum_dict, decile_filters, t, x, y, \
                                                        z, w, ss_qss, ss_wrf)

    for i in range(1, n_bins+1):
        nconc_i = get_bin_nconc(i, input_vars, rho_air)
        f_i = get_bin_vent_coeff(i, eta, N_Be_div_r3, N_Bo_div_r2, N_P, \
                                                    pres, rho_air, temp)
        spectrum_dict = add_dsd_data(spectrum_dict, env_filter, \
                                decile_filters, i-1, nconc_i, f_i)

    #close files for memory
    input_file.close()

    dsds_by_decile_filename = DATA_DIR + 'dsds_by_decile_' + \
                                        case_label + '_data.npy'

    np.save(dsds_by_decile_filename, spectrum_dict)

    return spectrum_dict

def get_decile_filters(ss_wrf):

    q = np.arange(0, 101, 10)
    decile_vals = np.percentile(ss_wrf, q)
    n_per_decile = int(np.shape(ss_wrf)[0]/10)
    rand_inds = np.random.choice(n_per_decile, n_spectra, replace=False)
    decile_filters = []

    for i, lo_bound in enumerate(decile_vals[:-1]):
        up_bound = decile_vals[i+1]
        decile_filter = np.logical_and.reduce(( \
                            (ss_wrf >= lo_bound), \
                            (ss_wrf < up_bound)))
        decile_filter = select_random_sample(decile_filter, rand_inds)
        decile_filters.append(decile_filter)

    return decile_filters, rand_inds

def select_random_sample(decile_filter, rand_inds):

    true_val_count = -1
    for i, val in enumerate(decile_filter):
        if val:
            true_val_count += 1
            if true_val_count not in rand_inds:
                decile_filter[i] = False

    return decile_filter

def add_non_dsd_data(spectrum_dict, decile_filters, t, x, y, z, \
                                                w, ss_qss, ss_wrf):

    for decile_filter in decile_filters:
        spectrum_dict['t_inds'].append(t[decile_filter])
        spectrum_dict['x_inds'].append(x[decile_filter])
        spectrum_dict['y_inds'].append(y[decile_filter])
        spectrum_dict['z_inds'].append(z[decile_filter])
        spectrum_dict['w_vals'].append(w[decile_filter])
        spectrum_dict['ss_qss_vals'].append(ss_qss[decile_filter])
        spectrum_dict['ss_wrf_vals'].append(ss_wrf[decile_filter])

    return spectrum_dict 

def add_dsd_data(spectrum_dict, env_filter, decile_filters, i, nconc_i, f_i):

    for j, decile_filter in enumerate(decile_filters):
        decile_key = 'decile_' + str(j+1)
        spectrum_dict[decile_key]['nconc'][i] = \
                                nconc_i[env_filter][decile_filter]
        spectrum_dict[decile_key]['vent_coeff'][i] = \
                                f_i[env_filter][decile_filter]

    spectrum_dict[decile_key]['nconc'] = \
        np.array(spectrum_dict[decile_key]['nconc'])
    spectrum_dict[decile_key]['vent_coeff'] = \
        np.array(spectrum_dict[decile_key]['vent_coeff'])

    return spectrum_dict

def make_and_save_dsds_by_decile_graphs(case_label, spectrum_dict):

    for i in range(10):
        decile_key = 'decile_' + str(i+1)
        fig, ax = plt.subplots()
        decile_spectra = get_decile_spectra(spectrum_dict[decile_key])
        for j, spectrum in enumerate(decile_spectra):
            ax.plot(1.e6*bin_radii, spectrum, c=colors_arr[j])

        outfile = FIG_DIR + 'dsds_by_decile_' + str(i+1) + '_' \
                                    + case_label + '_figure.png'
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()    

def get_decile_spectra(decile_dsd_dict):

    dsd = decile_dsd_dict['vent_coeff']*decile_dsd_dict['nconc']
    dsd = np.transpose(dsd)
    for i, row in enumerate(dsd):
        dsd = row*bin_radii/dlogDp

    return dsd

if __name__ == "__main__":
    main()
