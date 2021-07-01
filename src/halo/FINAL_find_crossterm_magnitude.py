"""
Print data summarizing effect of neglecting cross terms to calculate average
quasi-steady-state SS
"""
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.ss_functions import get_ss_qss_vs_t, get_lwc_vs_t, \
                                    get_full_spectrum_dict, \
                                    get_ss_qss_components

lwc_filter_val = 1.e-4
w_cutoff = 1

rmax = 102.e-6

change_CAS_corr = True
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
    
    ADLR_file = DATA_DIR + 'npy_proc/ADLR_alldates.npy'
    ADLR_dict = np.load(ADLR_file, allow_pickle=True).item()
    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    full_spectrum_dict = get_full_spectrum_dict(CAS_dict, \
                                CIP_dict, change_CAS_corr)

    lwc = get_lwc_vs_t(ADLR_dict, full_spectrum_dict, cutoff_bins, rmax)
    w = ADLR_dict['data']['w']
    ss_qss = get_ss_qss_vs_t(ADLR_dict, full_spectrum_dict, change_CAS_corr, \
                                cutoff_bins, full_ss, incl_rain, incl_vent)
    A, B, meanr, nconc = get_ss_qss_components(ADLR_dict, full_spectrum_dict, \
            change_CAS_corr, cutoff_bins, full_ss, incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    w = w[filter_inds]

    print(np.shape(ss_qss))
    print(np.nanmean(ss_qss))
    print(np.mean(A*w/(4*np.pi*B))*np.mean(1./(meanr*nconc))*100.)
    print(np.mean(A*w/(4*np.pi*B))/np.mean((meanr*nconc))*100.)
    print(np.mean(A*w/(4*np.pi*B)))
    print(np.mean((meanr*nconc)))

if __name__ == "__main__":
    main()
