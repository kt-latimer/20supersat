"""
altitude vs time, using 'above sea level' and 'pressure' altitude quantities from ADLR
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

figdirname = '/home/klatimer/proj/20supersat/figures/halo'

def main():
    """
    main routine.
    """
    data = np.load('/home/klatimer/proj/20supersat/data/halo/npy_proc/ADLR_20141001.npy', allow_pickle=True).item()    
    
    plt.subplot(2, 1, 1)
    plt.plot(data['data']['time'], data['data']['alt_asl'])
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('"Above sea level" altitude')

    plt.subplot(2, 1, 2)
    plt.plot(data['data']['time'], data['data']['alt_pres'])
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('"Pressure" altitude')

    outfile = figdirname + '/altvst_figure.png'
    plt.savefig(outfile)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
