"""
This is a package. (link to submodules with \:mod\:`<module_name>')
"""
BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/wrf/'
FIG_DIR = BASE_DIR + 'figures/wrf/'

n_WRF_bins = 33
WRF_bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(n_WRF_bins)]) #bin diams in m
WRF_bin_radii = bin_diams/2.
WRF_bin_dlogDp = np.array([np.log10(2.**(1./3.)) for i in range(n_WRF_bins)])
