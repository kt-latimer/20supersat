"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revhalo import DATA_DIR, FIG_DIR, PCASP_bins
from revhalo.ss_qss_calculations import get_lwc_from_cas

#for plotting
versionstr = 'v4_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'polluted': '#87cfeb', 'unpolluted': '#0b0bf7'}

#lwc_filter_val = 1.e-5 #v1-3
lwc_filter_val = 10 #v4
n_samples = -1 #v1, v4
#n_samples = 60 #v2 
#n_samples = 1 #v3

change_cas_corr = False
cutoff_bins = False 
incl_rain = False
incl_vent = False

#diams and differentials
diams = (PCASP_bins['upper'] + PCASP_bins['lower'])/2.
diam_diffl = (PCASP_bins['upper'] - PCASP_bins['lower'])
n_bins = np.shape(diams)[0]

#fan C_PI cutoff
unpoll_cutoff = 18 

def main():

    fan_aero_size_distb = get_fan_aero_size_distb()
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    aero_size_distb_alldates = None

    for date in good_dates:
        if date == '20140906' or date == '20140921': #dates dne for pcasp
            continue

        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        pcaspfile = DATA_DIR + 'npy_proc/PCASP_' + date + '.npy'
        pcasp_dict = np.load(pcaspfile, allow_pickle=True).item()

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        
        filter_inds = np.logical_and.reduce((
                        (lwc < lwc_filter_val), \
                        (temp > 273)))

        print(np.sum(filter_inds))
        print(np.shape(filter_inds))

        aero_size_distb = get_aero_size_distb(pcasp_dict['data'], \
                                                filter_inds, n_samples)

        aero_size_distb_alldates = add_to_alldates_array(aero_size_distb, \
                                    aero_size_distb_alldates)

        make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
                                            aero_size_distb, date, n_samples)

    make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
                                        aero_size_distb_alldates, \
                                        'alldates', n_samples)

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

def get_aero_size_distb(pcasp_data, filter_inds, n_samples):
        
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
        dN = np.array([np.nanmean( \
                pcasp_data[var_key][filter_inds][j:j+n_samples]) \
                for j in range(np.sum(filter_inds)) if j%n_samples == 0])
        dNdlogDp = 1./2.3025*dN*diams[i]/diam_diffl[i] # num/m^3
        aero_size_distb[:, i] = dNdlogDp
    
    return aero_size_distb
    
def make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
                                        aero_size_distb, date, n_samples):

        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.plot(diams*1.e9, 1.e-6*aero_size_distb.T, \
                linewidth=3)
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
                        + ', change_cas_corr=' + str(change_cas_corr) \
                        + ', cutoff_bins=' + str(cutoff_bins) \
                        + ', n samples per curve=' + str(n_samples))
        ax.legend()
        outfile = FIG_DIR + versionstr + 'aero_size_distb_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

if __name__ == "__main__":
    main()
