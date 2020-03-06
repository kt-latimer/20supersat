"""
Compare supersaturation calulated using CC empirical fit vs output value from
WRF ('SSW')
"""
from datetime import datetime

from netCDF4 import MFDataset
import numpy as np

from halo.utils import linregress

DATA_DIR = \
'/clusterfs/stratus/dromps/wrf_fan2018/Supersaturation/Supersaturation/'
SET_DIR = 'C_BG/'

def main():
    """
    the main routine.
    """
    #stats params
    n_samps = 100000
    null_m = 1.

    #physical constants
    C_ap = 1005. #dry air heat cap at const P (J/(kg K))
    D = 0.23e-4 #diffus coeff water in air (m^2/s)
    g = 9.8 #grav accel (m/s^2)
    L_v = 2501000. #latent heat of evaporation of water (J/kg)
    Mm_a = .02896 #Molecular weight of dry air (kg/mol)
    Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
    R = 8.317 #universal gas constant (J/(mol K))
    R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
    R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K)
   
    #wrf reference values
    P_0 = 1.e4 #ref pressure (Pa)
    th_0 = 300. #ref pot temp in (K)
   
    #load datafiles
    ncfile = MFDataset(DATA_DIR + SET_DIR + 'wrfout_d01_2014*', 'r')
    ncvars = ncfile.variables

    #get relevant variables from wrf output
    P_base = ncvars['PB'][...]
    P_pertb = ncvars['P'][...]
    q = ncvars['QVAPOR'][...] 
    SS_wrf = ncvars['SSW'][...]
    th_pertb = ncvars['T'][...]

    #convert wrf outputs to real quantities
    th = th_0 + th_pertb
    P = P_base + P_pertb
    T = th*np.power((P/P_0), R_a/C_ap) #temperature (K)
    
    #Fan's supersaturation calculation
    #saturation water vapor pressure (Pa)
    e_sat = 611.2*np.exp(17.67*np.add(-273.15, T)/np.add(-29.65, T))
    #saturation water vapor mixing ratio
    q_sat = (Mm_v/Mm_a)*e_sat/np.add(P, -1.*e_sat) 
    SS_fan = np.add(-1, q/q_sat)
    
    #get subset of entire dataset and run linear regression/stat analysis
    [SS_wrf_samp, SS_fan_samp], samp_inds = get_samples([SS_wrf, SS_fan], n_samps)
    m, b, R, p = linregress(SS_wrf_samp, SS_fan_samp)
    t_score = do_t_test(SS_wrf_samp, SS_fan_samp, m, b, null_m)

    #get environmental vars corresponding to SS samples
    P_samp, q_samp, T_samp = from_samp_inds([P, q, T], samp_inds)

    #output data to file
    datadict = {'inds': samp_inds, 'q':q_samp, 'P':P_samp, \
                'regression stats':{'b':b, 'm':m, 'p':p, 'R':R, \
                't_score':t_score}, 'SS_fan':SS_fan_samp, \
                'SS_wrf':SS_wrf_samp, 'T':T_samp}
    timestamp = datetime.timestamp(datetime.now())
    filename = DATA_DIR + SET_DIR + 'SS_sample_data_' + timestamp + '.npy'
    np.save(filename, datadict) 

def get_samples(full_sets, n_samps):
   """
   given array of datasets, take n_samps number of samples, selected randomly
   from all dimensions of the data (function assumes all elements of full_sets
   have the same shape). Output sample subsets as well as their indices in the
   original sets (same indices used for all original sets)
   """
    dims = np.shape(full_sets[0])
    n_dims = len(dims)

    samp_inds = np.empty((n_samps, n_dims))

    for i in range(n_dims):
        randarr = np.random.randint(dims[i], size=(n_samps, 1))
        randarr = randarr.transpose()
        samp_inds[:, i] = randarr

    samp_sets = [np.empty((n_samps, 1) for i in range(len(full_sets))]
    for i, row in enumerate(samp_inds):
        for j in range(len(samp_sets)):
            samp_sets[j][i, 1] = full_sets[j][tuple(row.astype(int))]
        
    return samp_sets, samp_inds

def do_t_test(x, y, m, b, null_m):
    """
    Student\'s t-test to determine t-score of regression slope m against the 
    null hypothesis null_m
    """

    return t_score

def from_samp_inds(full_sets, samp_inds)
    """
    given indices samp_inds and array of data sets full_sets, returns elements
    of each data set specified by the indices (same indices for all sets)
    """
    samp_sets = [np.empty((n_samps, 1) for i in range(len(full_sets))]
    for i, row in enumerate(samp_inds):
        for j in range(len(samp_sets)):
            samp_sets[j][i, 1] = full_sets[j][tuple(row.astype(int))]

    return samp_sets

if __name__ == "__main__":
    main()
