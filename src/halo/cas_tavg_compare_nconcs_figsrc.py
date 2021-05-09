"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
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

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    nconc_cas_alldates = None
    nconc_cip_alldates = None

    for date in good_dates:
        nconc_cas, nconc_cip = get_tavg_nconc_data(date)
        nconc_cas_alldates = add_to_alldates_array(nconc_cas, nconc_cas_alldates)
        nconc_cip_alldates = add_to_alldates_array(nconc_cip, nconc_cip_alldates)
        #make_and_save_nconc_scatter(nconc_cas, nconc_cip, date)

    x = nconc_cas_alldates
    y = nconc_cip_alldates
    make_and_save_nconc_scatter(nconc_cas_alldates, \
                    nconc_cip_alldates, 'alldates')

def add_to_alldates_array(nconc, nconc_alldates):

    if nconc_alldates is None:
        return nconc 
    else:
        return np.concatenate((nconc_alldates, nconc))

def get_tavg_nconc_data(date):

    print(date)
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    adlr_dict = get_tavg_dict(adlr_dict)
    cas_dict = get_tavg_dict(cas_dict)
    cip_dict = get_tavg_dict(cip_dict)

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 27.3), \
                            (w < 1000)))

    cas_t = cas_dict['data']['time'][filter_inds]
    nconc_cas = np.zeros(np.shape(cas_t))

    xi = cas_dict['data']['xi'][filter_inds]
    PAS = cas_dict['data']['PAS'][filter_inds]
    TAS = cas_dict['data']['TAS'][filter_inds]

    volume_corr_factor = xi/(PAS/TAS)

    for i in range(12, 17):
        var_name = 'nconc_' + str(i)
        nconc_i = cas_dict['data'][var_name][filter_inds]*volume_corr_factor
        nconc_cas += nconc_i

    nconc_cip = cip_dict['data']['nconc_1'][filter_inds]
    naninds = ~np.logical_or(np.isnan(nconc_cas), np.isnan(nconc_cip))
    infinds = ~np.logical_or(np.isinf(nconc_cas), np.isinf(nconc_cip))
    inds = np.logical_and(naninds, infinds)

    return nconc_cas[inds], nconc_cip[inds] 

def get_tavg_dict(data_dict):

    for key in data_dict['data'].keys():
        data_arr = data_dict['data'][key]
        new_data_arr = []
        i = 10
        while i < np.shape(data_arr)[0]:
           new_data_arr.append(np.nanmean(data_arr[i-10:i]))
           i += 1
        data_dict['data'][key] = np.array(new_data_arr)
    
    return data_dict 

def make_and_save_nconc_scatter(nconc_cas, nconc_cip, date):

    nconc_cas = nconc_cas*1.e-6
    nconc_cip = nconc_cip*1.e-6
    diff = nconc_cas - nconc_cip
    print(np.mean(diff))
    print(np.std(diff))

    m, b, R, sig = linregress(nconc_cas, nconc_cip)

    fig, ax = plt.subplots()

    ax.scatter(nconc_cas, nconc_cip)
    ax.set_xlabel('nconc from CAS bins 12-16 (cm$^{-3}$)')
    ax.set_ylabel('nconc from CIP bin 1 (cm$^{-3}$)')

    regline = b + m*nconc_cas
    conf_band = get_conf_band(nconc_cas, nconc_cip, regline) 
    ax.scatter(nconc_cas, regline + conf_band, c='b', alpha=0.4)
    ax.scatter(nconc_cas, regline - conf_band, c='b', alpha=0.4)
    #ax.fill_between(nconc_cas, regline + conf_band, \
    #                regline - conf_band, color='b', \
    #                alpha=0.4, label='95% confidence band')

    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c='black', \
            linestyle='dashed', \
            linewidth=2, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle=':', c='r')

    #ax.set_xlim([0, 55])
    #ax.set_ylim([0, 55])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()

    outfile = FIG_DIR + 'cas_tavg_compare_nconcs_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    
    
    fig, ax = plt.subplots()
    ax.hist(diff, bins=30)
    ax.set_yscale('log')

    outfile = FIG_DIR + 'cas_tavg_compare_nconcs_hist_' + date + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

def get_conf_band(xvals, yvals, regline):

    t = 2.131 #two-tailed 95% CI w/ 17 pts, 2 params --> 15 dof
    n_pts = 17 #lazy :D
    meanx = np.mean(xvals)
    x_quad_resid = (xvals - meanx)**2.
    y_quad_resid = (yvals - regline)**2. 
    se_pt = np.sqrt(np.sum(y_quad_resid)/(n_pts - 2)) 
    se_line = se_pt*np.sqrt(1./n_pts + x_quad_resid/np.sum(x_quad_resid))
    conf_band = t*se_line

    return conf_band

if __name__ == "__main__":
    main()
