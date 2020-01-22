"""
Create and save figure rhoairvst.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 22})

def main():
    """
    plots air density vs time and saves to figure directory 
    """
    
    date = '20141001'

    R_a = 287 #ideal gas const dry air
    
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlrdata = np.load(adlrfile, allow_pickle=True).item()

    t = adlrdata['data']['time']    
    #calculate air density using ideal gas law
    rhoair = adlrdata['data']['stat_pres']/(R_a*adlrdata['data']['stat_temp'])
    
    fig, ax = plt.subplots()
    ax.plot(t, rhoair)
   
    fig.set_size_inches(21, 12)
    
    outfile = FIG_DIR + 'rhoairvst_figure.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
