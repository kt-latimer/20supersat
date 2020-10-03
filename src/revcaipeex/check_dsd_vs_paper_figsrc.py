"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from revcaipeex import BASE_DIR, DATA_DIR, FIG_DIR, CDP_bins

#for plotting
colors = ['#2c2e35', '#ff441c', '#3fb84c', '#1c2694', \
            '#30c1b6', '#c853a9', '#ffea48', '#7d7d1d', \
            '#112694', '#862996', '#b54d4a', '#14a248'] 
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'font.family': 'serif'})

#timestamps
daytimes = {'20090616': [95551, 94533, 93933, 93209, 92835, \
             92131, 91639, 91433, 91127, 90810, 90502], \
            '20090621': [91622, 92540, 93843, 94059, 94545, \
             94754, 95007, 95241, 95851, 100125, 100158], \
            '20090622': [90247, 91610, 93327, 93707, 94331, \
             95133, 95628, 95702, 95852, 100124, 100302, 101742]}

#diams only for cloud droplets
center_diams = (CDP_bins['upper'][:30] + CDP_bins['lower'][:30])/2.
diffl_diam_um = (CDP_bins['upper'][:30] - CDP_bins['lower'][:30])*1.e6

def main():
    """
    for each date and for all dates combined, create and save w histogram.
    """
    for i, date in enumerate(daytimes.keys()):
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)

        #get cpd data and create new file with lwc entry
        filename = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        dataset = np.load(filename, allow_pickle=True).item()

        targettimes = daytimes[date]
        lines = {}

        for j, label in enumerate(dataset['data']['time']):
            if np.floor(label) in targettimes:
                nconc = np.array([dataset['data']['nconc_'+str(k)][j] \
                                                for k in range(1, 31)])
                dNdD = 1.e-6*nconc/diffl_diam_um #units cm^-3 um^-1
                lines.update({label: dNdD})
        
        for j, label in enumerate(lines.keys()):
            if date == '20090616':
                ax.plot(center_diams*1.e6, lines[label], color=colors[10-j], \
                        linewidth=6, marker=None, label=label)
            else:
                ax.plot(center_diams*1.e6, lines[label], color=colors[j], \
                        linewidth=6, marker=None, label=label)

        ax.set_xlabel(r'D ($\mu$m)')
        ax.set_xlim([2, 110])
        ax.set_xscale('log')
        ax.set_ylabel(r'dN/dD (cm$^{-3}$ $\mu$m$^{-1}$)')
        ax.set_yscale('log')
        if date == '20090616':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], loc=1)
        else:
            ax.legend(loc=1)
        ax.set_title(date)
        outfile = FIG_DIR + versionstr + 'check_dsd_vs_paper_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
