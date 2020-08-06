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
                        pad_lwc_arrays, linregress

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
        print(date)
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata, True, True)
        cdpdata = pad_lwc_arrays(cdpdata, True, True)

        #loop through reasonable time offset range ($\pm$ 9 sec)
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(casdata['data']['time']), \
            np.around(cdpdata['data']['time'])])
        datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)
        lwc_cas_ames = casdata['data']['lwc_calc'][casinds]/1.e6 #convert roughly from
                                                                 #kg/m3 to g/g
        #for j, val in enumerate(lwc_cas_ames):
        #    if val > 0.0008:
        #        print(casdata['data']['time'][casinds]

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]
        lwc_cas_ames = lwc_cas_ames[goodrows]
        print(lwc_cas_ames.shape)

        #filter on LWC vals
        lwc_cas = []
        for booleanind in range(4):
            lwc_cas.append(datablock[:, high_bin_cas+booleanind])

        if i == 0:
            lwc_cas_alldates = lwc_cas
            lwc_cas_ames_alldates = lwc_cas_ames
        else:
            for j in range(4):
                lwc_cas_alldates[j] = np.concatenate((lwc_cas_alldates[j], lwc_cas[j]))
                lwc_cas_ames_alldates[j] = \
                            np.concatenate((lwc_cas_ames_alldates[j], lwc_cas_ames[j]))

        #make histogram
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(21, 21)
        ax[0][0].scatter(lwc_cas_ames, lwc_cas[0])
        ax[0][1].scatter(lwc_cas_ames, lwc_cas[1])
        ax[1][0].scatter(lwc_cas_ames, lwc_cas[2])
        ax[1][1].scatter(lwc_cas_ames, lwc_cas[3])
        for n in range(2):
            for m in range(2):
                ax[n][m].set_xlim([0, 1.e-3])
        outfile = FIG_DIR + versionstr + 'lwc_from_ames_cas_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

    #make histogram
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(21, 21)
    ax[0][0].scatter(lwc_cas_ames_alldates, lwc_cas_alldates[0])
    ax[0][1].scatter(lwc_cas_ames_alldates, lwc_cas_alldates[1])
    ax[1][0].scatter(lwc_cas_ames_alldates, lwc_cas_alldates[2])
    ax[1][1].scatter(lwc_cas_ames_alldates, lwc_cas_alldates[3])
    outfile = FIG_DIR + versionstr + 'lwc_from_ames_cas_alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)

if __name__ == "__main__":
    main()
