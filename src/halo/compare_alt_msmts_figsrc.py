"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        pad_lwc_arrays, linregress, get_full_ss_vs_t

#for plotting
#colors = {'control': '#777777', 'modified': '#88720A'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

def main():
    """
    for each date and for all dates combined, create and save w histogram.
    """

    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
         '20140919', '20140918', '20140921', '20140927', '20140928', \
         '20140930', '20141001']
    
    for i, date in enumerate(dates):
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        alt_1 = adlrdata['data']['alt_pres']
        alt_2 = adlrdata['data']['alt_asl']
        alt_3 = adlrdata['data']['alt_igi']
        altmin = np.min(alt_1)
        altmax = np.max(alt_1)
        bounds = np.array([altmin, altmax])

        #make histogram
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(36, 12)
        ax[0].scatter(alt_1, alt_2)#, color=colors['tot_derv'])
        ax[0].plot(bounds, bounds, '--')
        ax[1].scatter(alt_2, alt_3)#, color=colors['tot_derv'])
        ax[1].plot(bounds, bounds, '--')
        ax[2].scatter(alt_3, alt_1)#, color=colors['tot_derv'])
        ax[2].plot(bounds, bounds, '--')
        ax[0].set_xlabel('Pressure altitude (m)')
        ax[0].set_ylabel('HASL altitude (m)')
        ax[1].set_xlabel('HASL altitude (m)')
        ax[1].set_ylabel('IGI altitude (m)')
        ax[2].set_xlabel('IGI altitude (m)')
        ax[2].set_ylabel('Pressure altitude (m)')
        outfile = FIG_DIR + versionstr + 'compare_alt_msmts' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
