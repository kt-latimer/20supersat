"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_ss_vs_t, get_lwc

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink =  colors_arr[3]

lwc_filter_val = 1.e-4
w_cutoff = 2

ss_min = -10
ss_max = 10 
z_min = -100
z_max = 6500

cutoff_bins = True
incl_rain = False 
incl_vent = False
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    ss_qss_dict = {'allpts': None, 'up10perc': None}
    z_dict = {'allpts': None, 'up10perc': None}
    z_bins_dict = {'Polluted': None, 'Unpolluted': None}

    for date in good_dates:
        ss_qss, up10perc_ss_qss, z, up10perc_z = get_ss_qss_and_z_data(date)

        if np.shape(ss_qss)[0] != 0:
            ss_qss_dict['allpts'] = add_to_alldates_array(ss_qss, \
                                            ss_qss_dict['allpts'])
            ss_qss_dict['up10perc'] = add_to_alldates_array(up10perc_ss_qss, \
                                            ss_qss_dict['up10perc'])

            z_dict['allpts'] = add_to_alldates_array(z, \
                                            z_dict['allpts'])
            z_dict['up10perc'] = add_to_alldates_array(up10perc_z, \
                                            z_dict['up10perc'])

    make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict['allpts'], \
            z_dict['allpts'], 'allpts')
    make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict['up10perc'], \
            z_dict['up10perc'], 'up10perc')

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def get_ss_qss_and_z_data(date):

    metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
    met_dict = np.load(metfile, allow_pickle=True).item()
    cpdfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cpd_dict = np.load(cpdfile, allow_pickle=True).item()

    lwc = get_lwc(cpd_dict,cutoff_bins)
    temp = met_dict['data']['temp']
    w = met_dict['data']['w']
    z = met_dict['data']['alt']
    ss_qss = get_ss_vs_t(met_dict, cpd_dict, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)

    #there's a weird outlier which the third line removes
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (ss_qss < 100), \
                    (temp > 273)))

    if np.sum(filter_inds) != 0:
        w_filt = w[filter_inds]
        up10perc_cutoff = np.percentile(w_filt, 90)
        up10perc_inds = np.logical_and.reduce((
                                (filter_inds), \
                                (w > up10perc_cutoff)))

        up10perc_ss_qss = ss_qss[up10perc_inds]
        ss_qss = ss_qss[filter_inds]
        print(date)
        #print(up10perc_ss_qss)
        #print(ss_qss)
        print()

        up10perc_z = z[up10perc_inds]
        z = z[filter_inds]
    else:
        ss_qss = np.array([])
        up10perc_ss_qss = np.array([])
        z = np.array([])
        up10perc_z = np.array([])

    return ss_qss, up10perc_ss_qss, z, up10perc_z

def make_and_save_bipanel_ss_qss_vs_z(ss_qss, z, label):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(18, 12)

    h_z, z_bins = np.histogram(z, bins=30, density=True)

    avg_ss_qss, avg_z, se = get_avg_ss_qss_and_z(ss_qss, z, z_bins)
    #ax1.fill_betweenx(avg_z, avg_ss_qss + se, avg_ss_qss - se, \
    #                                color=magma_pink, alpha=0.4)
    ax1.plot(avg_ss_qss, avg_z, linestyle='', marker='o', \
            color=magma_pink, linewidth=6)
    ax2.hist(z, bins=z_bins, density=True, orientation='horizontal', \
            facecolor=(0, 0, 0, 0.0), edgecolor=magma_pink, \
            histtype='stepfilled', linewidth=6, linestyle='-')

    #formatting
    ax1.set_xlim((ss_min, ss_max))
    ax1.set_ylim((z_min, z_max))
    ax1.yaxis.grid()
    ax2.set_ylim((z_min, z_max))
    ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax2.set_xlabel(r'$\frac{dn_{points}}{dz}$ (m/s$^{-1}$)')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax2.xaxis.set_major_formatter(formatter)

    ##custom legend
    #poll_line = Line2D([0], [0], color=magma_pink, \
    #                    linewidth=6, linestyle='-')
    #unpoll_line = Line2D([0], [0], color=magma_pink, \
    #                    linewidth=6, linestyle='--')
    #ax2.legend([poll_line, unpoll_line], ['Polluted', 'Unpolluted'])

    outfile = FIG_DIR + versionstr + 'FINAL_bipanel_ss_qss_vs_z_' \
            + label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_avg_ss_qss_and_z(ss_qss, z, z_bins):

    avg_ss_qss = np.zeros(np.shape(z_bins)[0] - 1)
    avg_z = np.zeros(np.shape(z_bins)[0] - 1)
    se = np.zeros(np.shape(z_bins)[0] - 1) #standard error

    for i, val in enumerate(z_bins[:-1]):
        lower_bin_edge = val
        upper_bin_edge = z_bins[i+1]

        bin_filter = np.logical_and.reduce((
                        (z > lower_bin_edge), \
                        (z < upper_bin_edge)))

        n_in_bin = np.sum(bin_filter)
        if n_in_bin == 0:
            avg_ss_qss[i] = np.nan
            se[i] = np.nan
            avg_z[i] = np.nan
        else:
            ss_qss_slice = ss_qss[bin_filter]
            print(i)
            print(ss_qss_slice)
            z_slice = z[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_slice)
            se[i] = np.nanstd(ss_qss_slice)/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_qss, avg_z, se

if __name__ == "__main__":
    main()
