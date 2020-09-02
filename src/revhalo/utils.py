"""
various utilities for revhalo module
"""
import numpy as np

#bin size data and settings depending on cutoff_bins param
#(indices are for columns of datablock variable)
casbinfile = DATA_DIR + 'CAS_bins.npy'
CAS_bins = np.load(casbinfile, allow_pickle=True).item()
centr_cas = (CAS_bins['upper'] + CAS_bins['lower'])/4. #diam to radius
dr_cas = CAS_bins['upper'] - CAS_bins['lower']
nbins_cas = len(centr_cas)

cdpbinfile = DATA_DIR + 'CDP_bins.npy'
CDP_bins = np.load(cdpbinfile, allow_pickle=True).item()
centr_cdp = (CDP_bins['upper'] + CDP_bins['lower'])/4. #diam to radius
dr_cdp = CDP_bins['upper'] - CDP_bins['lower']
nbins_cdp = len(centr_cdp)

cipbinfile = DATA_DIR + 'CIP_bins.npy'
CIP_bins = np.load(cipbinfile, allow_pickle=True).item()
centr_cip = (CIP_bins['upper'] + CIP_bins['lower'])/4. #diam to radius
dr_cip = CIP_bins['upper'] - CIP_bins['lower']
nbins_cip = len(centr_cip)
