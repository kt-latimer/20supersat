"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
from caipeex.utils import get_ss_full, get_meanr, get_nconc, linregress

#for scatterting
colors = {'ss': '#88720A'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

def main():
    """
    for each date and for all dates combined, create and save w histogram.
    """
    files = [f for f in listdir(DATA_DIR + 'npy_proc/')]
    used_dates = []
    i = 0
    for f in files:
        #get flight date and compare if already processed
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
        nconc_calc = metdata['data']['nconc_cdp']
        nconc = get_nconc(dataset)
        m, b, R, sig = linregress(nconc, nconc_calc)
        print(m, b, R**2)

        if i == 0:
            nconc_calc_alldates = nconc_calc
            nconc_alldates = nconc
        else:
            nconc_calc_alldates = np.concatenate((nconc_calc_alldates, nconc_calc))
            nconc_alldates = np.concatenate((nconc_alldates, nconc))

        xmin = np.min(nconc)
        xmax = np.max(nconc)
        ymin = np.min(nconc_calc)
        ymax = np.max(nconc_calc)

        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.scatter(nconc, nconc_calc)#, color=colors['tot_derv'])
        ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
                label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_title(date)
        ax.set_xlabel('my nconc (m^-3)')
        ax.set_ylabel('cdp nconc (m^-3)')
        ax.legend(loc=1)
        outfile = FIG_DIR + versionstr + 'nconc_compare_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

    #make histogram
    m, b, R, sig = linregress(nconc_alldates, nconc_calc_alldates)
    print(m, b, R**2)
    xmin = np.min(nconc_alldates)
    xmax = np.max(nconc_alldates)
    print(nconc_alldates.shape)
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.scatter(nconc_alldates, nconc_calc_alldates)#, color=colors['tot_derv'])
    ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.set_title(date)
    ax.set_xlabel('my nconc (m^-3)')
    ax.set_ylabel('cdp nconc (m^-3)')
    ax.legend(loc=1)
    outfile = FIG_DIR + versionstr + 'nconc_compare_alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
