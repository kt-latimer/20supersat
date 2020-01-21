"""
This is a module.
"""
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

def match_time_inds(t_arr1, t_arr2):
    """
    synchronize two arrays of time values. currently just rounds.

    Returns: (time_inds1, time_inds2)
    
    e.g. call data1[time_inds1] and data2[time_inds2] to get \
    synchronized sets from data1 and data2.
    """
    n1 = len(t_arr1)
    n2 = len(t_arr2)
    if n1 < n2:
        t_short = np.around(t_arr1)
        t_long = np.around(t_arr2)
    else:
        t_short = np.around(t_arr2)
        t_long = np.around(t_arr1)
    inds_short = []
    inds_long = []
    long_search_start_ind = 0
    for i1, t1 in enumerate(t_short):
        for i2, t2 in enumerate(t_long[long_search_start_ind:-1]):
            if t1 == t2:
                inds_short.append(i1)
                inds_long.append(i2)
                long_search_start_ind = i2
                break
    if n1 < n2:
        return (inds_short, inds_long)
    else:
        return(inds_long, inds_short)

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
    (set_t_inds, env_t_inds) = match_time_inds(setdata['time'], envdata['time'])

    return (rho_wat[set_t_inds]/rho_air[env_t_inds], np.array(set_t_inds))
