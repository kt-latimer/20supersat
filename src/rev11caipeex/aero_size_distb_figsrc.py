"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from rev11caipeex import DATA_DIR, FIG_DIR, DMA_bins, PCASP_bins

#for plotting
versionstr = 'v5_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'polluted': '#87cfeb', 'unpolluted': '#0b0bf7'}

n_samples = -1 #v1
#n_samples = 1 #v2; note each point is a 60-sec avg in raw data files

#diams and differentials
DMA_diams = (DMA_bins['upper'] + DMA_bins['lower'])/2.
DMA_diam_diffl = (DMA_bins['upper'] - DMA_bins['lower'])
n_dma_bins = np.shape(DMA_diams)[0]

PCASP_diams = (PCASP_bins['upper'] + PCASP_bins['lower'])/2.
PCASP_diam_diffl = (PCASP_bins['upper'] - PCASP_bins['lower'])
n_pcasp_bins = np.shape(PCASP_diams)[0]

#fan C_PI cutoff
unpoll_cutoff = 18 

def main():

    fan_aero_size_distb = get_fan_aero_size_distb()
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    dma_size_distb_alldates = None
    pcasp_size_distb_alldates = None

    for date in good_dates:

        dmafile = DATA_DIR + 'npy_proc/DMA_' + date + '.npy'
        dma_dict = np.load(dmafile, allow_pickle=True).item()
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        pcaspfile = DATA_DIR + 'npy_proc/PCASP_' + date + '.npy'
        pcasp_dict = np.load(pcaspfile, allow_pickle=True).item()

        dma_t = dma_dict['data']['time']
        met_t = met_dict['data']['time']
        pcasp_t = pcasp_dict['data']['time']
        
        met_temp = met_dict['data']['temp']
        pcasp_temp = get_pcasp_temp(met_t, pcasp_t, met_temp)
        dma_temp = get_dma_temp(dma_t, met_t, pcasp_temp, pcasp_t)

        dma_filter_inds = dma_temp > 273
        pcasp_filter_inds = pcasp_temp > 273

        print('dma')
        dma_size_distb = get_aero_size_distb(dma_dict['data'], \
                                            dma_filter_inds, n_samples, \
                                            n_dma_bins, DMA_diams,
                                            DMA_diam_diffl)
        print('pcasp')
        pcasp_size_distb = get_aero_size_distb(pcasp_dict['data'], \
                                            pcasp_filter_inds, n_samples, \
                                            n_pcasp_bins, PCASP_diams,
                                            PCASP_diam_diffl)

        dma_size_distb_alldates = add_to_alldates_array(dma_size_distb, \
                                    dma_size_distb_alldates)
        pcasp_size_distb_alldates = add_to_alldates_array(pcasp_size_distb, \
                                    pcasp_size_distb_alldates)

        make_and_save_aero_size_distb_plot(fan_aero_size_distb, dma_size_distb, \
                                            pcasp_size_distb, date, n_samples)

    make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
                                        dma_size_distb_alldates, \
                                        pcasp_size_distb_alldates, \
                                        'alldates', n_samples)

def get_dma_temp(dma_t, met_t, pcasp_temp, pcasp_t):

    dma_temp = []

    for i, t in enumerate(dma_t[:-1]):
        t_start = t
        t_end = dma_t[i+1]
        (i_start, i_end) = \
            get_pcasp_interval_for_dma_timestep(t_start, t_end, pcasp_t)
        dma_temp.append(np.mean(pcasp_temp[i_start:i_end]))

    last_t_start = dma_t[-1]
    last_t_end = last_t_start + 4*3600 #idk if interval is exactly 4 min but we
    (last_i_start, last_i_end) = \
        get_pcasp_interval_for_dma_timestep(last_t_start, last_t_end, pcasp_t)
    dma_temp.append(np.mean(pcasp_temp[last_i_start:last_i_end]))

    return np.array(dma_temp)

def get_pcasp_interval_for_dma_timestep(t_start, t_end, pcasp_t):

    for i, t in enumerate(pcasp_t):
        if t == t_start:
            i_start = i
            break

    for i, t in enumerate(pcasp_t[i_start:]):
        if t == t_end:
            i_end = i
            return (i_start, i_end)

    i_end = np.shape(pcasp_t)[0]
    return (i_start, i_end)

def get_pcasp_temp(met_t, pcasp_t, met_temp):

    pcasp_temp = []
    for i, t in enumerate(met_t):
        if t in pcasp_t:
            pcasp_temp.append(met_temp[i])

    return np.array(pcasp_temp)

def get_fan_aero_size_distb():

    #get aerosol psd extracted from fig S5C of fan et al 2018
    fan_aero_size_distb = []
    with open(DATA_DIR + 'fan_aero_size_distb.csv', 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            fan_aero_size_distb.append(row)
    fan_aero_size_distb = np.array(fan_aero_size_distb)
    fan_aero_size_distb = \
        fan_aero_size_distb[fan_aero_size_distb[:,0].argsort()]

    return fan_aero_size_distb

def add_to_alldates_array(aero_size_distb, aero_size_distb_alldates):

    if aero_size_distb_alldates is None:
        return aero_size_distb
    else:
        return np.concatenate((aero_size_distb_alldates, aero_size_distb))

def get_aero_size_distb(aero_data, filter_inds, n_samples, \
                        n_bins, diams, diam_diffl):

    # n_samples = # consecutive data pts to average in a row per asd curve
    # if n_samples is negative then a single curve (average over all pts)
    # is plotted
    if n_samples > 0:
        n_curves = ceil(np.sum(filter_inds)/n_samples)
    else:
        n_curves = 1
        n_samples = np.sum(filter_inds)

    aero_size_distb = np.zeros((n_curves, n_bins))

    for i in range(n_bins):
        var_key = 'nconc_' + str(i+1)
        #var_key = 'dNdlogDp_' + str(i+1)
        dN = np.array([np.nanmean( \
                aero_data[var_key][filter_inds][j:j+n_samples]) \
                for j in range(np.sum(filter_inds)) if j%n_samples == 0])
        dNdlogDp = 1./2.3025*dN*diams[i]/diam_diffl[i] # num/m^3
        #dNdlogDp = dN # num/m^3
        aero_size_distb[:, i] = dNdlogDp
    
    return aero_size_distb
    
def make_and_save_aero_size_distb_plot(fan_aero_size_distb, dma_size_distb, \
                                        pcasp_size_distb, date, n_samples):

        print(date)
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.plot(PCASP_diams*1.e9, (1./2.3025)*1.e-6*pcasp_size_distb.T, \
                linewidth=3)
        plt.gca().set_prop_cycle(None)
        ax.plot(DMA_diams*1.e9, 1.e-6*dma_size_distb.T, \
                linewidth=3, linestyle='--')
        #ax.plot(PCASP_diams*1.e9, 1.e2/(PCASP_diam_diffl/PCASP_diams), \
                #label='pcasp dlogdp')
        #ax.plot(DMA_diams*1.e9, 1.e2/(DMA_diam_diffl/DMA_diams), \
                #label='dma dlogdp')
        ax.set_xscale('log')
        ax.plot(np.power(10, fan_aero_size_distb[:, 0]), \
                fan_aero_size_distb[:, 1], \
                label='Fan WRF simulation - polluted case', \
                color=colors['polluted'], \
                linewidth=6)
        ax.plot(np.power(10, fan_aero_size_distb[unpoll_cutoff:, 0]), \
                fan_aero_size_distb[unpoll_cutoff:, 1], \
                label='Fan WRF simulation - unpolluted case', \
                color=colors['unpolluted'], \
                linewidth=6, \
                linestyle='--')
        ax.set_xlabel('diam (nm)')
        ax.set_ylabel('dN/dlogD (cm^-3)')
        ax.set_title(date + ' aerosol size distb' \
                          + ', n samples per curve=' + str(n_samples))
        #quick lazy dirty
        if n_samples == -1 and date == 'alldates':
            handles, labels = ax.get_legend_handles_labels()
            labels = ['20111024 - PCASP', '20111027 - PCASP', \
            '20111024 - DMA', '20111027 - DMA'] + labels
            ax.legend(labels)
        else:
            ax.legend()
        outfile = FIG_DIR + versionstr + 'aero_size_distb_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

if __name__ == "__main__":
    main()
