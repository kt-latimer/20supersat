"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
from caipeex.utils import get_ss_full, get_meanr, get_nconc

#for plotting
colors = {'bulk': '#095793', 'edge': '#88720A'}
versionstr = 'v3_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-5
w_cutoff = 2

cutoff_bins = False

def main():
    """
    for each date and for all dates combined, create and save w histogram.
    """
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

        #relevant phys qtys
        alt = metdata['data']['alt']
        lwc = dataset['data']['lwc_cloud']
        temp = metdata['data']['temp']
        time = metdata['data']['sectime']#in seconds
        ss_qss = get_ss_full(dataset, metdata, cutoff_bins)
        w = metdata['data']['vert_wind_vel']

        #set up arrays
        nperlayer = []
        nedgeperlayer = []
        lwcfifthperc = []
        ssbulk = []
        ssedge = []
        layermax = 500 #m

        ##for taking 5th perc of entire set
        #total_filter = np.logical_and.reduce((
        #                (lwc > lwc_filter_val), \
        #                (w > w_cutoff), \
        #                (w < 100), \
        #                (temp > 273)))
        #
        #total_cutoff = np.percentile(lwc[total_filter], 5)
        #bulk_filter = lwc[total_filter] >= total_cutoff
        #edge_filter = np.logical_not(bulk_filter)
        #ssbulk = ss_qss[total_filter][bulk_filter]
        #ssedge = ss_qss[total_filter][edge_filter]
        #group data in 500m layers by altitude
        while layermax < np.max(alt):
            #at 10:19:45 on 08/23 w=327m/s
            layer_filter = np.logical_and.reduce((
                            (lwc > lwc_filter_val), \
                            (layermax-500 <= alt), \
                            (alt < layermax), \
                            (w > w_cutoff), \
                            (w < 100), \
                            (temp > 273)))
            nedge = 0
            if np.sum(layer_filter) != 0:
                nperlayer.append(np.sum(layer_filter))
                perc_cutoff = np.percentile(lwc[layer_filter], 5)
                lwcfifthperc.append(perc_cutoff)
                for j, val in enumerate(ss_qss[layer_filter]):
                    if lwc[layer_filter][j] < perc_cutoff:
                        nedge += 1
                        ssedge.append(val)
                    else:
                        ssbulk.append(val)
            else:
                nperlayer.append(0)
                lwcfifthperc.append(np.nan)
            nedgeperlayer.append(nedge)
            layermax += 500
        
        print(nperlayer)
        print(nedgeperlayer)
        print(lwcfifthperc)
        print(ssedge)
        
        if i == 0:
            ssbulk_alldates = ssbulk
            ssedge_alldates = ssedge
        else:
            ssbulk_alldates = np.concatenate((ssbulk_alldates, ssbulk))
            ssedge_alldates = np.concatenate((ssedge_alldates, ssedge)) 

        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(ssbulk_alldates, bins=30, color=colors['bulk'], \
                alpha=0.7, label='bulk', density=True)
        ax.hist(ssedge_alldates, bins=30, color=colors['edge'], \
                alpha=0.7, label='edge', density=True)
        ax.set_title('SS distribution, LWC > 1.e-5 g/g, T > 273K, w > 2 m/s')
        ax.set_xlabel('SS (%)')
        #ax.set_ylabel('Count')
        ax.set_ylabel('Probability')
        ax.legend(loc=1)
        outfile = FIG_DIR + versionstr + 'ss_bulkedge_hist_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

    #make histogram
    nbulk = np.sum(ssbulk_alldates >= 2)
    nedge = np.sum(ssedge_alldates >= 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(ssbulk_alldates, bins=30, color=colors['bulk'], \
            alpha=0.7, label='bulk', density=True)
    ax.hist(ssedge_alldates, bins=30, color=colors['edge'], \
            alpha=0.7, label='edge', density=True)
    ax.text(0.8, 0.8, '$N_{SS>2\%, bulk}$: ' + str(nbulk), transform=ax.transAxes) 
    ax.text(0.8, 0.7, '$N_{SS>2\%, edge}$: ' + str(nedge), transform=ax.transAxes) 
    ax.set_title('SS distribution, LWC > 1.e-5 g/g, T > 273K, w > 2 m/s')
    ax.set_xlabel('SS (%)')
    #ax.set_ylabel('Count')
    ax.set_ylabel('Probability')
    ax.legend(loc=1)
    outfile = FIG_DIR + versionstr + 'ss_bulkedge_hist_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
