"""
Create and show figure rhoairvst.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 12})

def main():
    """
    plots air density vs time and opens in interactive plot
    """
    
    date = '20140911'
    
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlrdata = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    casdata = np.load(casfile, allow_pickle=True).item()

    adlrt = adlrdata['data']['time']    
    cast = casdata['data']['time']    
    #calculate air density using ideal gas law
    adlrtas = adlrdata['data']['TAS']
    castas = casdata['data']['TAS']    
    fig, ax = plt.subplots()
    ax.plot(adlrt, adlrtas, label='ADLR')
    ax.plot(cast, castas, label='CAS')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('Air speed (m/s)')
    fig.legend()
    
    plt.show()

if __name__ == "__main__":
    main()
