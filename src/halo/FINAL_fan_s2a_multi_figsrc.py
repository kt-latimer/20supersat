"""
Make scatter plots of max w vs aerosol nconc using various methods to determine
aerosol nconc. Save as separate figure files
"""
import csv
from math import ceil
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import sys

from halo import DATA_DIR, FIG_DIR
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]

nconc_methods = {'fan_table': 'Fan Table S1', \
                 'dig_plot': 'Fan Fig S2 (digitized)', \
                 'rusen_median': 'Rusen\'s median method', \
                 'rusen_mean': 'Rusen\'s mean method'}
p_vals = [0.1182, 0.1155, 0.4011, 0.3161]

def main():

    wmax = get_fan_s2a_multi_data('wmax')
    for i, method in enumerate(nconc_methods.keys()):
        method_label = nconc_methods[method]
        p_val = p_vals[i]
        nconc = get_fan_s2a_multi_data(method)
        m, b, R, sig = linregress(nconc, wmax)
        make_and_save_fan_s2a_multi_fig(nconc, wmax, m, b, R, \
                                    p_val, method, method_label)
    
def get_fan_s2a_multi_data(method):

    data = []
    
    filename = DATA_DIR + 'fan_fig_s2a_multi_' + method + '.csv'
    with open(filename, 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            data.append(row[0])

    data = np.array(data)

    return data 

def make_and_save_fan_s2a_multi_fig(nconc, wmax, m, b, R, \
                                p_val, method, method_label):

        fig, ax = plt.subplots()

        nconc, wmax = sort_arrays(nconc, wmax)

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
        ax.text(0.1, 0.8, 'p = ' + str(p_val), transform=ax.transAxes)
        ax.legend()

        fig.suptitle('Max vertical wind velocity versus aerosol number' \
                        + ' concentration \n (Number concentrations from ' \
                        + method_label + ')')

        outfile = FIG_DIR + 'FINAL_fan_fig_s2a_multi_' + method + '.png'
        plt.savefig(outfile, bbox_inches='tight')
        plt.close(fig=fig)    

def sort_arrays(nconc, wmax):

    ind_array = np.argsort(nconc)

    return nconc[ind_array], wmax[ind_array]

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
