"""
Calculate linear regression parameters for potential temperature reported
values from ADLR vs SHARC. (Just prints everything out)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock_with_sharc, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        pad_lwc_arrays, linregress

#for plotting
colors = {'lwc': '#777777', 'ss': '#88720A'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5

change_cas_corr = True
cutoff_bins = True

#make separate folder cause it's a shit ton of figures
FIG_DIR = FIG_DIR + 'edge_check/'
def main():
    """
    For each date with data from all three instruments, fit ADLR vs CAS SHARC 
    measured values, and for each value of ADLR/SHARC time offset, print out
    R^2, m, b.
    """

    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
         '20140919', '20140918', '20140921', '20140927', '20140928', \
         '20140930', '20141001']
    
    for i, date in enumerate(dates):
        print(date)
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        sharcfile = DATA_DIR + 'npy_proc/SHARC_' + date + '.npy'
        sharcdata = np.load(sharcfile, allow_pickle=True).item()

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata, change_cas_corr, cutoff_bins)

        #loop through reasonable time offset range ($\pm$ 9 sec)
        [adlrinds, casinds, sharcinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(casdata['data']['time']), \
            np.around(sharcdata['data']['time'])])
        datablock = get_datablock_with_sharc(adlrinds, casinds, sharcinds, \
                                    adlrdata, casdata, sharcdata)

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]

        #get time-aligned theta data 
        sharc_ss = datablock[:, -1]
        t_array = datablock[:, 0]
        w = datablock[:, 2]

        #filter on LWC vals
        booleanind = int(change_cas_corr) + int(cutoff_bins)*2
        lwc_cas = datablock[:, high_bin_cas+booleanind]
        #filter_inds = w > 2
        #filter_inds = lwc_cas > lwc_filter_val
        filter_inds = np.logical_and.reduce((
                        (lwc_cas > lwc_filter_val), \
                        (w > 2)))
        
        #for each point matching filtering criteria, create a dueal plot of lwc
        #and supersaturation within a time range of $\pm$ 15 seconds
        for j, row in enumerate(datablock):
            if filter_inds[j]:
                t_cent = row[0]
                (i_start, i_end) = get_ind_bounds(t_array, t_cent - 15, \
                                                    t_cent + 15)
                make_lwc_ss_subplot(t_array[i_start:i_end], \
                                    lwc_cas[i_start:i_end], \
                                    sharc_ss[i_start:i_end], \
                                    j, date)

def make_lwc_ss_subplot(t, lwc, ss, central_ind, date):
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.plot(t, lwc, color=colors['lwc'], label='LWC')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LWC (g/g)')

    ax2 = ax.twinx()
    ax2.plot(t, ss, color=colors['ss'], label='SS')
    ax2.plot([t[central_ind], t[central_ind]], ax2.get_ylim(), \
                linestyle='--', color='r')
    ax2.set_ylabel('SS (%)')
    
    lines, labels = ax.get_legend_handles_labels() 
    lines2, labels2 = ax2.get_legend_handles_labels() 
    
    ax.legend(lines + lines2, labels + labels2, loc=0)
    
    outfile = FIG_DIR + versionstr + 'edge_check_' \
            + date + '_' + str(t[central_ind]) + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
