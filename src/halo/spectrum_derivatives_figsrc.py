"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_spliced_cas_and_cip_dicts, \
                                get_spliced_cas_and_cip_bins, \
                                get_spliced_dlogDp

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_red = colors_arr[2]
magma_pink = colors_arr[5]
magma_orange = colors_arr[8]

splice_methods = ['cas_over_cip', 'cip_over_cas', 'wt_avg']
cas_max_inds = {'cas_over_cip': 17, 'cip_over_cas': 12, 'wt_avg': 12}

lwc_filter_val = 1.e-4 #10**(-3.5)
w_cutoff = 1

z_min = -100
z_max = 6500

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

##
## physical constants
##
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3) 

def main():
    
    for splice_method in splice_methods:
        print(splice_method)
        make_spectrum_derivative_plot(splice_method)

def make_spectrum_derivative_plot(splice_method):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 27.3), \
                            (w < 1000)))

    cas_dict, cip_dict = get_spliced_cas_and_cip_dicts(cas_dict, \
                        cip_dict, splice_method, change_cas_corr)

    CAS_bin_radii, CIP_bin_radii = get_spliced_cas_and_cip_bins(splice_method)
    bin_radii = np.concatenate((CAS_bin_radii, CIP_bin_radii))
    dlogDp = get_spliced_dlogDp(splice_method)
    derv_bin_radii = get_derv_bin_radii(bin_radii)
    derv_dlogDp = get_derv_dlogDp(bin_radii)

    time = cas_dict['data']['time']
    spectrum_derivative_vs_t = np.zeros(np.shape(derv_bin_radii))

    for i, t in enumerate(time):
        if filter_inds[i]:
            spectrum_derivative_at_t = get_spectrum_derivative_at_t(i, \
                    cas_dict, cip_dict, dlogDp, derv_dlogDp, splice_method)
            spectrum_derivative_vs_t = np.vstack((spectrum_derivative_vs_t, \
                                                spectrum_derivative_at_t))

    avg_spectrum_derivative = np.nanmean(spectrum_derivative_vs_t, axis=0)
    print(derv_bin_radii)
    print(avg_spectrum_derivative)
    std_spectrum_derivative = np.nanstd(spectrum_derivative_vs_t, axis=0)

    fig, ax = plt.subplots()

    ax.errorbar(derv_bin_radii[1:]*1.e6, \
                avg_spectrum_derivative[1:], \
                yerr=std_spectrum_derivative[1:], \
                color='grey', alpha=0.3, fmt='o')
    ax.step(derv_bin_radii[1:]*1.e6, \
            avg_spectrum_derivative[1:], \
            where='post', color=magma_pink)
    ax.plot([derv_bin_radii[-1]*1.e6, derv_bin_radii[-1]*1.e6], \
            [avg_spectrum_derivative[-1], \
            avg_spectrum_derivative[-1]], \
            color=magma_pink)

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\frac{d^2N(r)}{dlogD_p^2}$ (cm$^{-3}$)')
    #ax.set_yscale('log')

    outfile = FIG_DIR + 'spectrum_derivative_' + \
                    splice_method + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_derv_bin_radii(bin_radii):

    derv_bin_radii = []

    for i, r in enumerate(bin_radii[:-1]):
        derv_bin_radii.append(np.sqrt(r*bin_radii[i+1]))
    
    derv_bin_radii = np.array(derv_bin_radii)

    return derv_bin_radii

def get_spectrum_derivative_at_t(i, cas_dict, cip_dict, dlogDp, \
                                    derv_dlogDp, splice_method): 

    spectrum_at_t = get_spectrum(i, cas_dict, cip_dict, dlogDp, splice_method)

    spectrum_derivative_at_t = []

    for j, n in enumerate(spectrum_at_t[:-1]):
        derv = (spectrum_at_t[j+1] - n)/derv_dlogDp[j]
        spectrum_derivative_at_t.append(derv)

    return np.array(spectrum_derivative_at_t)

def get_spectrum(i, cas_dict, cip_dict, dlogDp, splice_method):

    spectrum = []
    j = 0

    for k in range(5, cas_max_inds[splice_method]):
        var_name = 'nconc_' + str(k)
        if change_cas_corr:
            var_name += '_corr'
        spectrum.append(cas_dict['data'][var_name][i]/dlogDp[j])
        j += 1

    for k in range(1, 20):
        var_name = 'nconc_' + str(k)
        spectrum.append(cip_dict['data'][var_name][i]/dlogDp[j])
        j += 1

    return np.array(spectrum)
        
def get_derv_dlogDp(bin_radii):

    derv_dlogDp = []

    for i, r in enumerate(bin_radii[:-1]):
        derv_dlogDp.append(np.log10(bin_radii[i+1]/r))
    
    derv_dlogDp = np.array(derv_dlogDp)

    return derv_dlogDp 

if __name__ == "__main__":
    main()
