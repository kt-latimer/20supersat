"""
various utility methods for halo package.
"""
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

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

low_bin_cas = 4 
high_bin_cas = low_bin_cas + nbins_cas
low_bin_cdp = high_bin_cas 
high_bin_cdp = low_bin_cdp + nbins_cdp

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

def linregress(x, y=None):
    """
    ~~copy pasta from scipy so I don't have to import the whole damn module~~
    Calculate a regression line
    This computes a least-squares regression for two sets of measurements.
    Parameters
    ----------
    x, y : array_like
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.
    Returns
    -------
    slope : float
        slope of the regression line
    intercept : float
        intercept of the regression line
    r-value : float
        correlation coefficient
    stderr : float
        Standard error of the estimate
    """
    TINY = 1.0e-20
    if y is None:  # x is a (2, N) or (N, 2) shaped array_like
        x = np.asarray(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            msg = "If only `x` is given as input, it has to be of shape (2, N) \
            or (N, 2), provided shape was %s" % str(x.shape)
            raise ValueError(msg)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    n = len(x)
    xmean = np.mean(x,None)
    ymean = np.mean(y,None)

    # average sum of squares:
    ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm*ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if (r > 1.0):
            r = 1.0
        elif (r < -1.0):
            r = -1.0

    df = n-2
    t = r*np.sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)))
    slope = r_num / ssxm
    intercept = ymean - slope*xmean
    sterrest = np.sqrt((1-r*r)*ssym / ssxm / df)
    return slope, intercept, r, sterrest

def get_datablock(adlrinds, casinds, cdpinds, adlrdata, casdata, cdpdata):
    """
    Consolidate data for easier processing 
    Format of output array (order of columns): time, temperature, vertical \
    velocity, cas correction factor,  number conc for cas bins (12 cols),\
     number conc for cdp bins (15 cols).
    """
    # extra four columns: time, temperature, 
    # vertical wind velocity, cas corr factor
    datablock = np.zeros([len(adlrinds), 4 + nbins_cas + nbins_cdp])
    datablock[:, 0] = np.fix(adlrdata['data']['time'][adlrinds])
    datablock[:, 1] = adlrdata['data']['stat_temp'][adlrinds]
    datablock[:, 2] = adlrdata['data']['vert_wind_vel'][adlrinds]
    datablock[:, 3] = casdata['data']['TAS'][casinds]\
			/casdata['data']['PAS'][casinds]\
			*casdata['data']['xi'][casinds]
    for i in range(nbins_cas):
        key = 'nconc_' + str(i+5)
        datablock[:, i+4] = casdata['data'][key][casinds]
    
    for i in range(nbins_cdp):
        key = 'nconc_' + str(i+1)
        datablock[:, i+4+nbins_cas] = cdpdata['data'][key][cdpinds]

    return datablock

def get_nconc_vs_t(datablock, change_cas_corr, cutoff_bins):
    """
    Returns (nconc_cas, nconc_cdp)
    """
    if cutoff_bins:
        cas_offset = 3
        cdp_offset = 2
    else:
        cas_offset = 0 
        cdp_offset = 0
    nconc_cas = []
    nconc_cdp = []
    for i, row in enumerate(datablock):
        if change_cas_corr:
            nconc_cas.append(np.sum(row[3]*row[(low_bin_cas+cas_offset):high_bin_cas]))
        else:
            nconc_cas.append(np.sum(row[(low_bin_cas+cas_offset):high_bin_cas]))
        nconc_cdp.append(np.sum(row[(low_bin_cdp+cdp_offset):high_bin_cdp]))
    return (np.array(nconc_cas), np.array(nconc_cdp))

def get_meanr_vs_t(datablock, change_cas_corr, cutoff_bins):
    """
    Returns (meanr_cas, meanr_cdp)
    """
    if cutoff_bins:
        cas_offset = 3
        cdp_offset = 2
    else:
        cas_offset = 0
        cdp_offset = 0
    meanr_cas = []
    meanr_cdp = []
    for row in datablock:
        if change_cas_corr:
            meanr_cas.append(np.sum(row[3]*row[(low_bin_cas+cas_offset):high_bin_cas]\
                *centr_cas[nbins_cas-(high_bin_cas-(low_bin_cas+cas_offset)):nbins_cas])\
                /np.sum(row[3]*row[(low_bin_cas+cas_offset):high_bin_cas]))
        else:
            meanr_cas.append(np.sum(row[(low_bin_cas+cas_offset):high_bin_cas]\
                *centr_cas[nbins_cas-(high_bin_cas-(low_bin_cas+cas_offset)):nbins_cas])\
                /np.sum(row[(low_bin_cas+cas_offset):high_bin_cas]))
        meanr_cdp.append(np.sum(row[(low_bin_cdp+cdp_offset):high_bin_cdp]\
            *centr_cdp[nbins_cdp-(high_bin_cdp-(low_bin_cdp+cdp_offset)):nbins_cdp])\
            /np.sum(row[(low_bin_cdp+cdp_offset):high_bin_cdp]))
    return (np.array(meanr_cas), np.array(meanr_cdp))
