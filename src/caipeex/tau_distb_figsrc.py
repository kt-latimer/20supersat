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
colors = {'tau': '#88720A'}
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
        #get flight date and check if already procetaued
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
        
        ss_qss = get_ss_full(dataset, metdata)
        ss_qss = ss_qss[filter_inds]
        outlier_filter = ss_qss > 100 #at 10:19:45 on 08/23 w=327m/s
        meanr = get_meanr(dataset)
        nconc = get_nconc(dataset)
        tau = 1./(meanr*nconc)
        tau = tau[filter_inds][np.logical_not(outlier_filter)]
        #print(np.mean(tau[filter_inds]), np.std(tau[filter_inds]), \
        #        np.median(tau[filter_inds]), np.min(tau[filter_inds]), \
        #        np.max(tau[filter_inds]))
        #print(np.mean(meanr[filter_inds]), np.std(meanr[filter_inds]), \
        #        np.min(meanr[filter_inds]), np.max(meanr[filter_inds]))
        #print(np.mean(nconc[filter_inds]), np.std(nconc[filter_inds]), \
        #        np.min(nconc[filter_inds]), np.max(nconc[filter_inds]))
        print()

        if i == 0:
            tau_alldates = tau
        else:
            tau_alldates = np.concatenate((tau_alldates, tau))
        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.hist(tau, bins=30)#, color=colors['tot_derv'])
        ax.set_title('tau distribution, LWC > 1.e-4 g/g, T > 273K, w > 2 m/s')
        ax.set_xlabel('tau (s)')
        ax.set_ylabel('Count')
        outfile = FIG_DIR + versionstr + 'tau_distb_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

    #make histogram
    print(tau_alldates.shape)
    high_tau_filter = tau_alldates > 1
    print(np.sum(high_tau_filter))
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.hist(tau_alldates, bins=30)#, color=colors['tot_derv'])
    ax.set_title('tau distribution, LWC > 1.e-4 g/g, T > 273K, w > 2 m/s')
    ax.set_xlabel('tau (s)')
    ax.set_ylabel('Count')
    outfile = FIG_DIR + versionstr + 'tau_distb_' \
            + 'alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
