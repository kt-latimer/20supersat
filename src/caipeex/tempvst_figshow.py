"""
Create and show figure rhoairvst.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 18})

def main():
    """
    plots air density vs time and opens in interactive plot
    """
    
    date = '20090621'

    #get met data for that date
    filename = DATA_DIR + 'npy_proc/MET_' + date + '.npy' 
    metdata = np.load(filename, allow_pickle=True).item()

    time = metdata['data']['sectime']#in seconds
    temp = metdata['data']['temp']
    
    fig, ax = plt.subplots()
    ax.plot(time, temp)
    
    plt.show()

if __name__ == "__main__":
    main()
