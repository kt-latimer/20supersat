"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
from caipeex.utils import get_ss_full, get_meanr, get_nconc, dsd_radii

#for plotting
colors = {'ss': '#88720A'}
versionstr = 'v2_'

FIG_DIR = FIG_DIR + 'dsd_set/'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

def main():
    """
    for each date and for all dates combined, create and save w histogram.
    """
    r = dsd_radii[:30] #in microns; for plotting
    dr = 
    files = [f for f in listdir(DATA_DIR + 'npy_proc/')]
    used_dates = []
    i = 0
    for f in files:
        #get flight date and check if already processed
        date = f[-12:-4]
        if date in used_dates:
            continue
        else:
            print(date)
            used_dates.append(date)
        
        #get met data for that date
        filename = DATA_DIR + 'npy_proc/MET_' + date + '.npy' 
        metdata = np.load(filename, allow_pickle=True).item()

        #get dsd data and create new file with lwc entry
        filename = DATA_DIR + 'npy_proc/DSD_' + date + '.npy'
        dataset = np.load(filename, allow_pickle=True).item()

        time = metdata['data']['sectime']#in seconds
        lwc = dataset['data']['lwc_cloud']
        w = metdata['data']['vert_wind_vel']
        temp = metdata['data']['temp']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))

        for val in filter_inds:
            if val is True:
                #r_pdf = 
                #nr_pdf = 
                #r = 
                t = time[filter_inds]
                fig, ax = plt.subplot(1, 2)
                ax[0].plot(r, r_pdf)
                ax[0].set_xlabel('radius (um)')
                ax[0].set_ylabel('pdf for nconc (m^-4)')
                ax[1].plot(r, nr_pdf)
                ax[1].set_xlabel('radius (um)')
                ax[1].set_ylabel('pdf for drop radius (m^-3)')
                fig.set_size_inches(21, 12)
                outfile = FIG_DIR + versionstr + 'dsd_set_' \
                        + date + '_' + str(t) + '_figure.png'
                plt.savefig(outfile)
                plt.close(fig=fig)

if __name__ == "__main__":
    main()
