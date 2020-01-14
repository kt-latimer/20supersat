"""
Plotting procedure based on https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/simple_plot.html
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

figdirname = '/home/klatimer/proj/20supersat/figures/caipeex'

def get_x():
    """
    Get x values for plotting
    """
    return np.arange(0.0, 2.0, 0.01)

def get_y():
    """
    Get y values for plotting
    """
    return 1 + np.sin(2 * np.pi * get_x())

def main():
    """
    Plots y versus x and saves to figure directory
    """
    fig, ax = plt.subplots()
    ax.plot(get_x(), get_y())
    
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
            title='About as simple as it gets, folks')
    ax.grid()

    outfile = figdirname + '/example_figure.png'
    fig.savefig(outfile)

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
