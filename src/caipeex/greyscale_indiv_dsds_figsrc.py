"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from os import listdir

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
from caipeex.utils import get_ss_full, get_meanr, get_nconc, \
                          centr_dsd, dr_dsd

#for plotting
hilite_colors = ['#332288', '#5A56B0', '#117733', '#44AA99', '#88CCEE', \
                    '#DDCC77', '#CC6677', '#AA4499', '#882255']
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

centr_dsd = centr_dsd[:30]
dr_dsd = dr_dsd[:30]

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

        lwc = dataset['data']['lwc_cloud']
        meanr = get_meanr(dataset, cutoff_bins)
        nconc = get_nconc(dataset, cutoff_bins)
        ss = get_ss_full(dataset, metdata, cutoff_bins)
        temp = metdata['data']['temp']
        time = dataset['data']['time']#in seconds
        w = metdata['data']['vert_wind_vel']

        filter_inds = np.logical_and.reduce((
                        (lwc > lwc_filter_val), \
                        (w > w_cutoff), \
                        (temp > 273)))
        hi_ss_inds = ss > 2
        
        N = np.sum(filter_inds)
        delta_rgb = 200./N
        greyscale_colors = [((20 + i*delta_rgb)/255., (20 + i*delta_rgb)/255., \
                            (20 + i*delta_rgb)/255.) for i in range(N)] 

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(24, 12)
        ngraphed = 0
        nhilited = 0
        for j, val in enumerate(filter_inds):
            if val:
                if not hi_ss_inds[j]:
                    #units cm^-3 um^-1
                    dNdr = 1.e-12*np.array([dataset['data']['nconc_'+str(k)][j] \
                                  for k in range(1, 31)])/dr_dsd #units cm^-3 um^-1
                    dNdlnr = dNdr*centr_dsd
                    ax[0].plot(centr_dsd*1.e6, dNdr, marker=None, \
                                linewidth=1, color=greyscale_colors[ngraphed])
                    ax[1].plot(centr_dsd*1.e6, dNdlnr, marker=None, \
                                linewidth=1, color=greyscale_colors[ngraphed])
                else:
                    #units cm^-3 um^-1
                    dNdr = 1.e-12*np.array([dataset['data']['nconc_'+str(k)][j] \
                                  for k in range(1, 31)])/dr_dsd #units cm^-3 um^-1
                    dNdlnr = dNdr*centr_dsd
                    label_j = str(time[j])# + ', ' + str(temp[j])
                    ax[0].plot(centr_dsd*1.e6, dNdr, marker=None, \
                                linewidth=1, color=greyscale_colors[ngraphed], \
                                path_effects=[pe.Stroke( \
                                    linewidth=4, \
                                    foreground=hilite_colors[nhilited]), \
                                    pe.Normal()], \
                                label=label_j, 
                                zorder=N-nhilited)
                    ax[1].plot(centr_dsd*1.e6, dNdlnr, marker=None, \
                                linewidth=1, color=greyscale_colors[ngraphed], \
                                path_effects=[pe.Stroke( \
                                    linewidth=4, \
                                    foreground=hilite_colors[nhilited]), \
                                    pe.Normal()], \
                                label=label_j, 
                                zorder=N-nhilited)
                    nhilited += 1
                ngraphed += 1

        ax[0].set_xlabel(r'r ($\mu$m)')
        ax[0].set_ylabel(r'dN/dr (cm$^{-3}$ $\mu$m$^{-1}$)')
        ax[0].legend(loc=1)
        ax[1].set_xlabel(r'r ($\mu$m)')
        ax[1].set_ylabel(r'dN/d(ln(r)) (cm$^{-3}$)')
        ax[1].legend(loc=1)
        fig.suptitle(date + ' LWC > 1.e-4 g/g, T > 273K, w > 2 m/s')

        outfile = FIG_DIR + versionstr + 'greyscale_indiv_dsds_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

if __name__ == "__main__":
    main()
