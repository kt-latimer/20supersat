"""
check bin size ranges.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

figdirname = '/home/klatimer/proj/20supersat/figures/halo'

def main():
    for i, setname in enumerate(['CAS', 'CDP', 'NIXECAPS']):
        bins = np.load('/home/klatimer/proj/20supersat/data/halo/' \
                + setname + '_bins.npy', allow_pickle=True).item()
        upper = bins['upper']
        lower = bins['lower']
        plt.subplot(1,3,i+1)
        plt.plot(upper, 'bo')
        plt.plot(lower, 'ro')
        plt.xlabel('bin number')
        plt.ylabel('bin diam (um)')
        plt.title(setname)

    outfile = figdirname + '/checkbins_figure.png'
    plt.savefig(outfile)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
