"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revhalo import DATA_DIR, FIG_DIR, PCASP_bins, UHSAS_bins, UHSAS2_bins
from revhalo.ss_qss_calculations import get_lwc_from_cas
from revhalo.aero_calculations import get_kernel_weighted_nconc

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

#lwc_filter_val = 10 #v1
#temp_filter_val = -273 #v1
lwc_filter_val = 1.e-5 #v2
temp_filter_val = 273 #v2
n_samples = -1 #v1,2

change_cas_corr = False
cutoff_bins = False 

#diams and differentials
bin_diams = {'PCASP': (PCASP_bins['upper'] + PCASP_bins['lower'])/2., \
         'UHSAS': (UHSAS_bins['upper'] + UHSAS_bins['lower'])/2., \
         'UHSAS2': (UHSAS2_bins['upper'] + UHSAS2_bins['lower'])/2.}
bin_diam_diffls = {'PCASP': (PCASP_bins['upper'] - PCASP_bins['lower']), \
                'UHSAS': (UHSAS_bins['upper'] - UHSAS_bins['lower']), \
                'UHSAS2': (UHSAS2_bins['upper'] - UHSAS2_bins['lower'])}
n_bins = {'PCASP': PCASP_bins['upper'].shape[0], \
          'UHSAS': UHSAS_bins['upper'].shape[0], \
          'UHSAS2': UHSAS2_bins['upper'].shape[0]}
bin_start_inds = {'PCASP': 1, 'UHSAS': 15, 'UHSAS2': 4}

kernels = {'PCASP': np.ones(n_bins['PCASP']), \
            'UHSAS': np.ones(n_bins['UHSAS']), \
            'UHSAS2': np.ones(n_bins['UHSAS2'])}

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    for date in good_dates:
        if date == '20140906' or date == '20140921': #dates dne for pcasp
            continue

        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        pcaspfile = DATA_DIR + 'npy_proc/PCASP_' + date + '.npy'
        pcasp_dict = np.load(pcaspfile, allow_pickle=True).item()
        uhsasfile = DATA_DIR + 'npy_proc/UHSAS_' + date + '.npy'
        uhsas_dict = np.load(uhsasfile, allow_pickle=True).item()

        uhsas_setname = get_uhsas_setname(date)

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        z = adlr_dict['data']['alt']
        
        filter_inds = np.logical_and.reduce((
                        (lwc < lwc_filter_val), \
                        (temp > temp_filter_val)))

        pcasp_nconc = get_kernel_weighted_nconc(pcasp_dict, \
                                                kernels['PCASP'], \
                                                n_bins['PCASP'], \
                                                bin_start_inds['PCASP'])
        uhsas_nconc = get_kernel_weighted_nconc(uhsas_dict, \
                                                kernels[uhsas_setname], \
                                                n_bins[uhsas_setname], \
                                                bin_start_inds[uhsas_setname])

        pcasp_nconc = pcasp_nconc[filter_inds]
        uhsas_nconc = uhsas_nconc[filter_inds]
        z = z[filter_inds]

        make_and_save_z_vs_nconc_plot(z, pcasp_nconc, uhsas_nconc, date)

def get_uhsas_setname(date):
    
    if date in ['20140916', '20140918', '20140919', '20140921']:
        return 'UHSAS2'
    else:
        return 'UHSAS'

def make_and_save_z_vs_nconc_plot(z, pcasp_nconc, uhsas_nconc, date):

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(21, 21)
        ax[0].scatter(pcasp_nconc*1.e-6, z, label='PCASP')
        ax[0].scatter(uhsas_nconc*1.e-6, z, label='UHSAS')
        ax[0].set_xlabel('Nconc (cm^-3)')
        ax[0].set_ylabel('Altitude (m)')
        ax[0].set_title(date + ' aerosol nconc vert distb' \
                        + ', change_cas_corr=' + str(change_cas_corr) \
                        + ', cutoff_bins=' + str(cutoff_bins))
        ax[0].legend()
        ax[1].hist(z, bins=30, orientation='horizontal')
        ax[1].set_xlabel('Count')
        ax[1].set_ylim(ax[0].get_ylim())
        outfile = FIG_DIR + versionstr + 'aero_z_vs_nconc_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

if __name__ == "__main__":
    main()
