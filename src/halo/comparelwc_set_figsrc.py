"""
Generates set of plots of LWC vs time from ADLR, CAS, and CDP instruments \
for all flight dates. Also look at effect of 3um diam bin cutoff and \
adjusting CAS particle correction factor after Weigel 2016. 
"""

from itertools import product
from os import listdir

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import calc_lwc

matplotlib.rcParams.update({'font.size': 22})

def main():
    """
    main routine; calls make_lwc_fig for all flight dates (even if datasets \
    are missing) and all combinations of boolean values for 3um bin cutoff \
    and cas correction factor change. 
    """
    setnames = ['CAS', 'CDP']

    files = [f for f in listdir(DATA_DIR + 'npy_proc/')]
    used_dates = []
    for f in files:
        #get flight date and check for existing figure
        date = f[-12:-4]
        if date in used_dates:
            continue
        else:
            print(date)
            used_dates.append(date)
        
        #try to get adlr data for that date. if it doesn't exist, don't \
        #proceed because we won't have sufficient environmental data
        try:
            filename = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy' 
            adlrdata = np.load(filename, allow_pickle=True).item()
        except FileNotFoundError:
            print(DATA_DIR + 'npy_proc/ADLR_' + date \
                + '.npy')
            print('No ADLR data for ' + date + '. No figure made.')
            continue

        #get all datasets corresponding to flight date
        existing_datasets = []
        existing_setnames = []
        for setname in setnames:
            try:    
                filename = DATA_DIR + 'npy_proc/' + setname + '_' + date + '.npy'
                dataset = np.load(filename, allow_pickle=True)
            except FileNotFoundError:
                print(filename + 'not found')
                continue
            existing_datasets.append(dataset.item())
            existing_setnames.append(setname)
            
            #make figure for all combinations of boolean params
            for cutoff_bins, change_cas_corr in product([True, False], repeat=2):
                make_lwc_figure(adlrdata, existing_datasets, \
                        existing_setnames, cutoff_bins, change_cas_corr, date)
    
    #make set summary figure (basically for make compatibility)
    plt.text(0, 4, 'second lwc compare figure set.')
    plt.text(0, 3, 'file name format:')
    plt.text(0, 2, '"v2comparelwc_<YYYYMMDD>"')
    plt.text(2, 1, '"<cutoff_bins>"')
    plt.text(2, 0, '"<change_cas_corr>"')
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())

    outfile = FIG_DIR + 'v3comparelwc_set_figure.png'
    plt.savefig(outfile)

def make_lwc_figure(adlrdata, datasets, setnames, cutoff_bins, change_cas_corr, date):
    """
    plot and save lwc comparison (ADLR, CAS, CDP) figure given numerical \
    data and flags to enforce 3um diameter minimum or uniform correction \
    of CAS and CDP data.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    
    booleankey = str(int(cutoff_bins)) + str(int(change_cas_corr))
    colors = {'ADLR': '#777777', 'CAS': '#95B9E9', 'CDP': '#FC6A0C'}
    
    #plot ADLR
    t_adlr = adlrdata['data']['time']
    lwc_adlr = adlrdata['data']['lwc']
    ax.plot(t_adlr, lwc_adlr, label='ADLR', color=colors['ADLR'])
    
    #get lwc for and plot CAS and/or CDP if available
    #plot CDP first for visualization since values are typically higher
    for setname in ['CDP', 'CAS'] :
        if setname in setnames:
            i = setnames.index(setname)
            dataset = datasets[i]
            lwc = dataset['data']['lwc'][booleankey]
            lwc_t_inds = dataset['data']['lwc_t_inds']
            t = dataset['data']['time'][lwc_t_inds]
            
            ax.plot(t, lwc, label=setname, color=colors[setname])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('LWC (g/g)')
    ax.set_ylim(0, 0.0001)

    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc=0)

    plt.title('Date: ' + date + ' | Cutoff at 3um diam: ' + str(cutoff_bins) \
            + ' | Same CAS/CDP corr factors: ' + str(change_cas_corr)) 

    outfile = FIG_DIR + 'v4comparelwc_' + date + str(cutoff_bins) + \
            str(change_cas_corr) + '.png'
    fig.savefig(outfile)
    plt.close()

#run main() if user enters 'python [module path].py' from command line
if __name__ == "__main__":
    main()
