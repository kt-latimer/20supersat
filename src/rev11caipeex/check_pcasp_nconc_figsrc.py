"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from rev11caipeex import DATA_DIR, FIG_DIR
from rev11caipeex.ss_qss_calculations import linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

n_samples = -1 #v1
#n_samples = 1 #v2; note each point is a 60-sec avg in raw data files

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    pcasp_nconc_alldates = None
    pcasp_nconc_from_file_alldates = None

    for date in good_dates:

        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        pcaspfile = DATA_DIR + 'npy_proc/PCASP_' + date + '.npy'
        pcasp_dict = np.load(pcaspfile, allow_pickle=True).item()

        met_t = met_dict['data']['time']
        pcasp_t = pcasp_dict['data']['time']

        pcasp_nconc_from_file = met_dict['data']['pcasp_nconc_from_file']
        met_t_inds = get_met_t_inds(met_t, pcasp_t)
        pcasp_nconc_from_file = pcasp_nconc_from_file[met_t_inds]

        pcasp_nconc = get_nconc_vs_t(pcasp_dict['data'])

        pcasp_nconc_from_file_alldates = \
            add_to_alldates_array(pcasp_nconc_from_file_alldates, \
                                    pcasp_nconc_from_file_alldates)
        pcasp_nconc_alldates = add_to_alldates_array(pcasp_nconc, \
                                    pcasp_nconc_alldates)

        make_and_save_nconc_scatter_plot(pcasp_nconc, \
                                        pcasp_nconc_from_file, date)

    make_and_save_nconc_scatter_plot(pcasp_nconc_alldates, \
                                        pcasp_nconc_from_file_alldates, \
                                        'alldates')

def get_met_t_inds(met_t, pcasp_t):

    met_t_inds = []

    for i, t in enumerate(met_t):
        if t in pcasp_t:
            met_t_inds.append(i)

    return met_t_inds

def get_nconc_vs_t(pcasp_data):

    pcasp_nconc = np.zeros(np.shape(pcasp_data['nconc_1']))

    for key in pcasp_data.keys():

        if 'nconc' in key:
            pcasp_nconc += pcasp_data[key]
            
    return np.array(pcasp_nconc)

def add_to_alldates_array(aero_size_distb, aero_size_distb_alldates):

    if aero_size_distb_alldates is None:
        return aero_size_distb
    else:
        return np.concatenate((aero_size_distb_alldates, aero_size_distb))

def make_and_save_nconc_scatter_plot(pcasp_nconc, pcasp_nconc_from_file, date):

        m, b, R, sig = linregress(pcasp_nconc, pcasp_nconc_from_file)
        print(m, b)

        print(date)
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.scatter(pcasp_nconc, pcasp_nconc_from_file)
        ax.set_xlabel('my PCASP nconc (m^-3)')
        ax.set_ylabel('PCASP nconc from file (m^-3)')
        outfile = FIG_DIR + versionstr + 'check_aero_nconc_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

if __name__ == "__main__":
    main()
