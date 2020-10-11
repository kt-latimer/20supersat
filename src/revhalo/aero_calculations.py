"""
various routines and subroutines for calculating quasi-steady-state
superaturation from HALO campaign data
*random and non-comprehensive notes*
-abandoning 'meanfr' notation and just putting boolean incl_vent arg
-option to calculate full ss_qss is also now a boolean arg
"""
import numpy as np
import re

from revhalo import PCASP_bins 

##
## center radii of bins
##
PCASP_bin_radii = (PCASP_bins['upper'] + PCASP_bins['lower'])/4.

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

def get_kernel_weighted_nconc(asd_dict, kernel, n_bins, bin_start_ind):

    kernel_weighted_nconc = np.zeros(np.shape(asd_dict['data']['time']))

    for i in range(n_bins):
        var_key = 'nconc_' + str(i+bin_start_ind)
        kernel_weighted_nconc += asd_dict['data'][var_key]*kernel[i]

    return kernel_weighted_nconc
