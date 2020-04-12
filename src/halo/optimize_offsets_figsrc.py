"""
plot correlation coeffs as a function of CAS/CDP time offset for each flight
date
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
colors = {'meanr': '#4A8CCA', 'nconc': '#FC6A0C'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})

def main():
    """
    plots R^2 vs delta t for each flight date and saves figure
    """
    
    #get cleaned up data (came from running optimal_offsets.py then I removed
    #some extra junk output - see notes)
    filename = DATA_DIR + 'v2_optimal_offsets_TT_lwcfilt_clean.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]

    #lay out plot
    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(16,16))
    rows = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    cols = [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    #loop through all dates and make plot
    lines_per_date = 19
    for i in range(int(len(lines)/lines_per_date)):
        start_ind = i*lines_per_date
        datestr = lines[start_ind][0]
        corr_data = lines[start_ind+1:start_ind+lines_per_date]
        offsets = [int(line[0]) for line in corr_data]
        nconc_rsq = [float(line[1]) for line in corr_data]
        meanr_rsq = [float(line[2]) for line in corr_data]

        ax[rows[i]][cols[i]].plot(offsets, nconc_rsq, \
                                    color=colors['nconc'], \
                                    linestyle='-', \
                                    marker='o',
                                    label='Number concentration')
        ax[rows[i]][cols[i]].plot(offsets, meanr_rsq, \
                                    color=colors['meanr'], \
                                    linestyle='-', \
                                    marker='o',
                                    label='Mean radius')
        ax[rows[i]][cols[i]].set_title(datestr)

        #if rows[i] == 3 and cols[i] == 1:
        #    ax[rows[i]][cols[i]].set_xlabel('CAS/CDP time offset (sec)')    

    plt.tight_layout()

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('CAS/CDP time offset (sec)')
    plt.ylabel('R^2')
    handles, labels = ax[3][2].get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')
    
    outfile = FIG_DIR + versionstr + 'optimize_offsets_figure.png'
    plt.savefig(outfile)

if __name__ == "__main__":
    main()
