"""
make and save histograms showing SS_QSS distribution from HALO CDP measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

change_cdp_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    nconc_cdp_alldates = None
    nconc_cip_alldates = None

    for date in good_dates:
        nconc_cdp, nconc_cip = get_nconc_data(date)
        nconc_cdp_alldates = add_to_alldates_array(nconc_cdp, nconc_cdp_alldates)
        nconc_cip_alldates = add_to_alldates_array(nconc_cip, nconc_cip_alldates)
        make_and_save_nconc_scatter(nconc_cdp, nconc_cip, date)

    inds = np.logical_and(np.logical_not(np.isnan(nconc_cdp_alldates)), \
                    np.logical_not(np.isnan(nconc_cip_alldates)))

    x = nconc_cdp_alldates[inds]
    y = nconc_cip_alldates[inds]
    inds = inds 
    #inds = x < 4.e6
    print(np.shape(inds))
    print(np.sum(x > y))
    print(np.sum(inds))

    #x = x[inds]
    #y = y[inds]

    make_and_save_nconc_scatter(x, y, 'alldates')
    #make_and_save_nconc_scatter(nconc_cdp_alldates[inds], nconc_cip_alldates[inds])

def add_to_alldates_array(nconc, nconc_alldates):

    if nconc_alldates is None:
        return nconc 
    else:
        return np.concatenate((nconc_alldates, nconc))

def get_nconc_data(date):

    print(date)
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdp_dict = np.load(cdpfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 27.3), \
                            (w < 1000)))

    cdp_t = cdp_dict['data']['time'][filter_inds]
    nconc_cdp = np.zeros(np.shape(cdp_t))

    for i in range(11, 16):
        var_name = 'nconc_' + str(i)
        nconc_i = cdp_dict['data'][var_name][filter_inds]
        nconc_cdp += nconc_i

    nconc_cip = cip_dict['data']['nconc_1'][filter_inds]

    return nconc_cdp, nconc_cip 

def make_and_save_nconc_scatter(nconc_cdp, nconc_cip, date):

    nconc_cdp = nconc_cdp*1.e-6
    nconc_cip = nconc_cip*1.e-6
    m, b, R, sig = linregress(nconc_cdp, nconc_cip)

    fig, ax = plt.subplots()

    ax.scatter(nconc_cdp, nconc_cip)
    ax.set_xlabel('nconc from CDP bins 11-15 (cm$^{-3}$)')
    ax.set_ylabel('nconc from CIP bin 1 (cm$^{-3}$)')
    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c='black', \
            linestyle='dashed', \
            linewidth=2, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.legend()

    outfile = FIG_DIR + 'cdp_compare_nconcs_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
