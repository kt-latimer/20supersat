"""
various utility methods for halo package.
"""
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

def get_ind_bounds(arr, minval, maxval, startind=0):
    """
    Return: (imin, imax) where arr[imin] >= minval and arr[imax] <= maxval.
    Assumes arr is sorted smallest to largest (ie time series)
    Starts sorting at startind in arr, if specified
    """
    i = startind
    while (arr[i] < minval) or (np.isnan(arr[i])):
        i += 1
    imin = i
    while (arr[i] < maxval) or (np.isnan(arr[i])):
        i += 1
    imax = i
    return(imin, imax)

def match_two_arrays(arr1, arr2):
    """
    Return: (inds1, inds2) where arr1[inds1] = arr2[inds2].
    Assumes arr1 and arr2 are both sorted in the same order (ie time series)
    """
    inds1 = []
    inds2 = []
    startind2 = 0
    for i1, x1 in enumerate(arr1):
        for i2, x2 in enumerate(arr2[startind2:]):
            if x1 == x2:
                inds1.append(i1)
                inds2.append(i2+startind2)
                startind2 = i2 + startind2 + 1
                break
    return(inds1, inds2)

def match_multiple_arrays(arrays):
    """
    Return: [inds1, ... , indsN] where arr1[inds1] = ... = arrN[indsN].
    Assumes all arrays are sorted in the same order (ie time series)
    probably a better way to do this recursively but I never learned that shit xd
    """
    inds = [[i for i in range(len(arrays[0]))]]
    for i, array in enumerate(arrays[:-1]):
        (inds1, inds2) = match_two_arrays([array[i] for i in inds[-1]], arrays[i+1])
        inds = [[indsj[i] for i in inds1] for indsj in inds]
        inds.append(inds2)
    return inds

def calc_lwc(setname, setdata, envdata, cutoff_bins, change_cas_corr):
    """
    calculate liquid water content given particle number concentrations \
    and environmental data. 

    Returns: (lwc_array, time_inds)
    -lwc_array = array of lwc values
    -time_inds = indices of setname array originally passed by user which \
    match indexing of lwc_array (i.e. lwc_array is time-commensurate with \
    setname[time_inds].
    
    update 1/20/20: if envdata is None-valued, return None's. For cleaner \
    data processing flow.
    """
    #return None values if envdata is None
    if envdata is None:
        return (None, None)

    #load particle diameters
    bin_dict = np.load(DATA_DIR + setname+ '_bins.npy', \
            allow_pickle=True).item()
    
    #constants
    rho_lw = 1.e3 #pure liquid water density
    R_a = 287 #ideal gas const dry air
    
    #calculate air density using ideal gas law
    rho_air = envdata['stat_pres']/(R_a*envdata['stat_temp'])
    print(np.nanmean(rho_air))

    #calculate density of liquid water in atmosphere by summing over bins
    n = len(setdata['time'])
    sum_nconc_radcubed = np.zeros((n,))
    print(setname, sum_nconc_radcubed.shape)
    i = 0
    for key in setdata.keys():
        if 'nconc' in key and 'tot' not in key:
            if cutoff_bins and bin_dict['lower'][i]<3.e-6:
                i += 1
                continue
            d_mean = (bin_dict['upper'][i] + bin_dict['lower'][i])/2.
            r = d_mean/2.
            if setname == 'CAS' and change_cas_corr:
                corr_factor = setdata['xi']*setdata['TAS']/setdata['PAS']
            else:
                corr_factor = np.ones((n,))
            sum_nconc_radcubed += np.power(r, 3.)*corr_factor*setdata[key]
            i += 1
    rho_wat = rho_lw*4./3.*np.pi*sum_nconc_radcubed

    #match time values so we can combine set data with environmental data
    #(currently just rounding)
    (set_t_inds, env_t_inds) = match_two_arrays( \
            np.around(setdata['time']), np.around(envdata['time']))
    print(len(set_t_inds), len(env_t_inds))
    return (rho_wat[set_t_inds]/rho_air[env_t_inds], np.array(set_t_inds))
