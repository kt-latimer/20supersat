"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys

from revhalo import DATA_DIR, FIG_DIR, PCASP_bins, UHSAS_bins, UHSAS2_bins
from revhalo.ss_qss_calculations import get_lwc_from_cas
from revhalo.utils import linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[3]

def main():

    nconc, wmax = get_fan_s2a_data()
    m, b, R, sig = linregress(nconc, wmax)
    make_and_save_fan_s2a_fig(nconc, wmax, m, b, R)
    
def get_fan_s2a_data():

    #get aerosol psd extracted from fig S2A of fan et al 2018
    nconc = []
    wmax = []

    with open(DATA_DIR + 'fan_fig_s2a.csv', 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            nconc.append(row[0])
            wmax.append(row[1])

    nconc = np.array(nconc)
    wmax = np.array(wmax)

    return nconc, wmax 

def make_and_save_fan_s2a_fig(nconc, wmax, m, b, R):

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 12)

        ax.scatter(nconc, wmax, color=magma_pink)

        ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
                c='k', \
                linestyle='dashed', \
                linewidth=3, \
                label=('m = ' + str(np.round(m, decimals=4)) + \
                        ', b = ' + str(np.round(b, decimals=4)) + \
                        ', R^2 = ' + str(np.round(R**2, decimals=4))))

        ax.set_xlabel(r'Aerosol concentration, D > 15nm (cm$^{-3}$)')
        ax.set_ylabel(r'w$_{max}$ (m/s)')
        fig.legend()
        outfile = FIG_DIR + versionstr + 'FINAL_fan_fig_s2a.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

if __name__ == "__main__":
    main()
