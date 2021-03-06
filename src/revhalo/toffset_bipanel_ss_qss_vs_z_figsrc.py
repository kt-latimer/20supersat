"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from revhalo import DATA_DIR, FIG_DIR
from revhalo.ss_qss_calculations import get_ss_vs_t_cas, get_lwc_from_cas

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 23})
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
colors_dict ={'allpts': colors_arr[3], 'up10perc': colors_arr[7]}

lwc_filter_val = 1.e-4
w_cutoff = 2

z_min = -100
z_max = 6500

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
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

    h_z, z_bins = np.histogram(z_dict['allpts'], bins=30, density=True)

    make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict, z_dict, z_bins)

def add_to_alldates_array(ss_qss, ss_qss_alldates):

    if ss_qss_alldates is None:
        return ss_qss
    else:
        return np.concatenate((ss_qss_alldates, ss_qss))

def get_ss_qss_and_z_data(date):

    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    adlr_dict, cas_dict, cip_dict = \
                sync_instrument_times(adlr_dict, cas_dict, cip_dict)

    lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']
    z = adlr_dict['data']['alt']
    ss_qss = get_ss_vs_t_cas(adlr_dict, cas_dict, cip_dict, \
                change_cas_corr, cutoff_bins, full_ss, \
                incl_rain, incl_vent)

    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
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

def sync_instrument_times(adlr_dict, cas_dict, cip_dict):

    delta_t_cas = get_cas_adlr_t_offset(adlr_dict, cas_dict) 

    adlr_dict, cas_dict, cip_dict = \
                align_all_vars(adlr_dict, cas_dict, cip_dict, \
                                adlr_inds, cas_inds, cip_inds)

    return adlr_dict, cas_dict, cip_dict

def align_all_vars(adlr_dict, cas_dict, cip_dict, \
                    adlr_inds, cas_inds, cip_inds):

    for key in adlr_dict.keys():
        adlr_dict[key] = adlr_dict[key][adlr_inds]

    for key in cas_dict.keys():
        cas_dict[key] = cas_dict[key][cas_inds]

    for key in cip_dict.keys():
        cip_dict[key] = cip_dict[key][cip_inds]

    return adlr_dict, cas_dict, cip_dict

def make_and_save_bipanel_ss_qss_vs_z(ss_qss_dict, z_dict, z_bins):

    fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(18, 12)

    for key in ss_qss_dict.keys():
        color = colors_dict[key]
        ss_qss = ss_qss_dict[key]
        z = z_dict[key]
        dz = np.array([z_bins[i+1] - z_bins[i] for i in \
                        range(np.shape(z_bins)[0] - 1)])
        print(key)

        avg_ss_qss, avg_z, se = get_avg_ss_qss_and_z(ss_qss, z, z_bins)
        notnan_inds = np.logical_not(np.isnan(avg_ss_qss))
        avg_ss_qss = avg_ss_qss[notnan_inds]
        avg_z = avg_z[notnan_inds]
        dz = dz[notnan_inds]

        #print(np.sum(avg_ss_qss*dz)/np.sum(dz))
        #continue

        ax1.plot(avg_ss_qss, avg_z, linestyle='-', marker='o', \
                color=color, linewidth=6, markersize=17)
        ax2.hist(z, bins=z_bins, density=True, orientation='horizontal', \
                facecolor=color, alpha=0.5)
                #facecolor=(0, 0, 0, 0.0), edgecolor=color, \
                #histtype='stepfilled', linewidth=6, linestyle='-')

    #formatting
    ax1.set_ylim((z_min, z_max))
    ax1.yaxis.grid()
    ax2.set_ylim((z_min, z_max))
    ax2.yaxis.grid()
    ax1.set_xlabel(r'$SS_{QSS}$ (%)')
    ax2.set_xlabel(r'$\frac{dn_{points}}{dz}$ (m$^{-1}$)')
    ax1.set_ylabel(r'z (m)')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax2.xaxis.set_major_formatter(formatter)

    #custom legend
    allpts_line = Line2D([0], [0], color=colors_dict['allpts'], \
                        linewidth=6, linestyle='-')
    up10perc_line = Line2D([0], [0], color=colors_dict['up10perc'], \
                        linewidth=6, linestyle='-')
    ax2.legend([allpts_line, up10perc_line], ['All cloudy updrafts', \
                                    'Top 10% cloudy updrafts (by w)'])

    outfile = FIG_DIR + versionstr + \
        'FINAL_combined_bipanel_ss_qss_vs_z_figure.png'
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
            z_slice = z[bin_filter]
            avg_ss_qss[i] = np.nanmean(ss_qss_slice)
            se[i] = np.nanstd(ss_qss_slice)/np.sqrt(np.sum(bin_filter))
            avg_z[i] = np.nanmean(z_slice)

    return avg_ss_qss, avg_z, se

if __name__ == "__main__":
    main()
