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
from revhalo.utils import linregress

#for plotting
versionstr = 'v2_'
#matplotlib.rcParams.update({'font.size': 21})
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
        #fig.set_size_inches(12, 12)

        ax.scatter(nconc, wmax, color=magma_pink)

        regline = b + m*np.array(nconc)
        ax.plot(ax.get_xlim(), b + m*np.array(ax.get_xlim()), \
                c='k', \
                linestyle='dashed', \
                linewidth=3, \
                label=('m = ' + str(np.round(m, decimals=4)) + \
                        ', b = ' + str(np.round(b, decimals=4)) + \
                        ', R^2 = ' + str(np.round(R**2, decimals=4))))
        conf_band = get_conf_band(nconc, wmax, regline) 
        ax.fill_between(nconc, regline + conf_band, \
                        regline - conf_band, color=magma_pink, \
                        alpha=0.4, label='95% confidence band')

        ax.set_xlabel(r'Aerosol concentration, D > 15nm (cm$^{-3}$)')
        ax.set_ylabel(r'w$_{max}$ (m/s)')
        ax.legend()

        fig.suptitle('Max vertical wind velocity versus aerosol number' \
                        + ' concentration \n (Reconstructed from [1])')

        outfile = FIG_DIR + versionstr + 'FINAL_fan_fig_s2a.png'
        plt.savefig(outfile, bbox_inches='tight')
        plt.close(fig=fig)    

def get_conf_band(xvals, yvals, regline):

    t = 2.131 #two-tailed 95% CI w/ 17 pts, 2 params --> 15 dof
    n_pts = 17 #lazy :D
    meanx = np.mean(xvals)
    x_quad_resid = (xvals - meanx)**2.
    y_quad_resid = (yvals - regline)**2. 
    se_pt = np.sqrt(np.sum(y_quad_resid)/(n_pts - 2)) 
    se_line = se_pt*np.sqrt(1./n_pts + x_quad_resid/np.sum(x_quad_resid))
    conf_band = t*se_line

    return conf_band

if __name__ == "__main__":
    main()
