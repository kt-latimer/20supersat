"""
plot my vs file SMPS total nconc to make sure everything is straight 
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os

from goama import DATA_DIR, FIG_DIR, SMPS_bins

#for plotting
versionstr = 'v3_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

# bin diams
smps_radii = np.sqrt(SMPS_bins['upper']*SMPS_bins['lower'])/2.
smps_dlogDp = np.log10(SMPS_bins['upper']/SMPS_bins['lower'])

def main():

    datafile = DATA_DIR + 'maoaossmpsS1.a1.20140326.000206.cdf'
    dataset = Dataset(datafile, 'r')

    dsd = dataset.variables['number_size_distribution']
    time = dataset.variables['time']

    above15conc = np.zeros(288)
    above50conc = np.zeros(288)

    for i, r in enumerate(smps_radii):
        if r > 7.5e-9:
            above15conc += smps_dlogDp[i]*dsd[:, i]
        if r > 25e-9:
            above50conc += smps_dlogDp[i]*dsd[:, i]

    timeavg = []
    above15nconcavg = []
    above50nconcavg = []

    i = 6
    while i < 288:
        timeavg.append(time[i])
        above15nconcavg.append(np.mean(above15conc[i-6:i]))
        above50nconcavg.append(np.mean(above50conc[i-6:i]))
        i += 1

    timeavg = np.array(timeavg)
    above15nconcavg = np.array(above15nconcavg)
    above50nconcavg = np.array(above50nconcavg)

    filter_inds = timeavg > 54000

    fig, ax = plt.subplots()
    ax.plot(timeavg[filter_inds], above15nconcavg[filter_inds], label='D>15')
    ax.plot(timeavg[filter_inds], above50nconcavg[filter_inds], label='D>50')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('aerosol concentration')
    ax.legend()

    outfile = FIG_DIR + versionstr + 'check_rusen_plot_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
