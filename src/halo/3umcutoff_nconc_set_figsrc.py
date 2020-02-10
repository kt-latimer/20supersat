"""
Create and save figure 3umcutoff_nconc_set.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

matplotlib.rcParams.update({'font.size': 22})

def main():
    """
    the main routine.
    """
    #possibly useful generic lines:
    #
    #fig, ax = plt.subplots()
    #...or:
    #fig=plt.gcf() 
    #
    #fig.set_size_inches(21, 12)
    outfile = FIG_DIR + '3umcutoff_nconc_set_figure.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
