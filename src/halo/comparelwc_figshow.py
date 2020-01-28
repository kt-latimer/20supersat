"""
plot lwc for given date and time range and show in interactive window
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 12})

def main():
    """
    creates and shows the figure
    """
    
    date = '20140911'
    
    casmin = 0 
    casmax = -1 
    cdpmin = 0 
    cdpmax = -1 
    
    #load all data sets
    adlrdatafile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlrdata = np.load(adlrdatafile, allow_pickle=True).item()
    
    casdatafile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    casdata = np.load(casdatafile, allow_pickle=True).item()
    caslwc = casdata['data']['lwc']['11']
    caslwctinds = casdata['data']['lwc_t_inds']
    cast = casdata['data']['time'][caslwctinds]

    cdpdatafile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdpdata = np.load(cdpdatafile, allow_pickle=True).item()
    cdplwc = cdpdata['data']['lwc']['11']
    cdplwctinds = cdpdata['data']['lwc_t_inds']
    cdpt = cdpdata['data']['time'][cdplwctinds]
    
    #plot shit
    fig, ax = plt.subplots()

    ax.plot(cast[casmin:casmax], caslwc[casmin:casmax])
    ax.plot(cdpt[cdpmin:cdpmax], cdplwc[cdpmin:cdpmax])

    #listen for user clicks 
    fig.canvas.mpl_connect('button_press_event',onclick)
    plt.show()

def onclick(event):
    """
    Deal with click events. Statements printed to terminal. [copied from \
    http://www.ster.kuleuven.be/~pieterd/python/html/plotting/interactive.html]
    """
    
    button = ['left','middle','right']  
    toolbar = plt.get_current_fig_manager().toolbar
    if toolbar.mode!='':
        print("You clicked on something, but toolbar \
                is in mode {:s}.".format(toolbar.mode))
    else:
        print("You {0}-clicked coords ({1},{2}) \
                (pix ({3},{4}))".format(button[event.button+1],\
                                                    event.xdata,\
                                                    event.ydata,\
                                                    event.x,\
                                                    event.y))
if __name__ == "__main__":
    main()
