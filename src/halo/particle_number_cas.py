"""
Convert CAS number concentrations to particle counts (based on formulae and
constants in Braga2017 and Weigel2016)
"""
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from utils import low_bin_cas, nbins_cas

#constants
A_s = 0.27e-6 #sample area (m^2)

def get_ptcl_num():
    """
    the main routine.
    """
    dates = ['20140906']
    for i, date in enumerate(dates):
        #load data
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        castas = casdata['data']['TAS']
        castime = casdata['data']['time']
        casdeltat = np.ediff1d(castime, to_begin=None, to_end=[0]) #1st elt = 0
        n = np.shape(casdata['data']['time'])[0]
        casnconc = np.empty((n, nbins_cas))
        for i in range(nbins_cas):
            key = 'nconc_' + str(i+5)
            casnconc[:, i] = casdata['data'][key]
        castas = np.reshape(castas, (n, 1))
        casdeltat = np.reshape(casdeltat, (n, 1))
       # print(casdeltat)
       # a = A_s*casnconc
       # b = a*castas
       # print(b)
       # c = b*casdeltat
        with np.errstate(invalid='ignore'):
            casptclnum = A_s*casnconc*castas#*casdeltat
   # return c
    return casptclnum

if __name__ == "__main__":
    get_ptcl_num()
