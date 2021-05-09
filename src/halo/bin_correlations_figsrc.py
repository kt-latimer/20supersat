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
magma_pink = colors_arr[5]

change_cas_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    rsq_vals = []
    x_labels = []

    for i in range(5, 16):
        rsq = make_adjacent_bin_plot(i, 'CAS')
        rsq_vals.append(rsq)
        x_labels.append('CAS bin ' + str(i) + ' vs ' + str(i+1))
        
    rsq = make_overlap_bin_plot()
    rsq_vals.append(rsq)
    x_labels.append('CAS bins 12-16 vs CIP bin 1')

    for i in range(1, 19):
        rsq = make_adjacent_bin_plot(i, 'CIP')
        rsq_vals.append(rsq)
        x_labels.append('CIP bin ' + str(i) + ' vs ' + str(i+1))

    make_rsq_chart(rsq_vals, x_labels)

def make_adjacent_bin_plot(i, instr_name):
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    nconc_bin1_alldates = None
    nconc_bin2_alldates = None

    for date in good_dates:
        nconc_bin1, nconc_bin2 = get_nconc_data(date, i, instr_name)
        nconc_bin1_alldates = add_to_alldates_array(nconc_bin1, nconc_bin1_alldates)
        nconc_bin2_alldates = add_to_alldates_array(nconc_bin2, nconc_bin2_alldates)

    rsq = make_and_save_nconc_scatter(nconc_bin1_alldates, \
                    nconc_bin2_alldates, i, instr_name)

    return rsq 

def make_overlap_bin_plot():
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    nconc_casbins_alldates = None
    nconc_cipbin_alldates = None

    for date in good_dates:
        nconc_casbins, nconc_cipbin = get_overlap_nconc_data(date)
        nconc_casbins_alldates = add_to_alldates_array(nconc_casbins, nconc_casbins_alldates)
        nconc_cipbin_alldates = add_to_alldates_array(nconc_cipbin, nconc_cipbin_alldates)

    rsq = make_and_save_nconc_scatter(nconc_casbins_alldates, \
                    nconc_cipbin_alldates, 543, 'both')

    return rsq 

def add_to_alldates_array(nconc, nconc_alldates):

    if nconc_alldates is None:
        return nconc 
    else:
        return np.concatenate((nconc_alldates, nconc))

def get_nconc_data(date, i, instr_name):

    #print(date)
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    datafile = DATA_DIR + 'npy_proc/' + instr_name + '_' + date + '.npy'
    data_dict = np.load(datafile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 27.3), \
                            (w < 1000)))

    var_name1 = 'nconc_' + str(i)
    var_name2 = 'nconc_' + str(i+1)

    if instr_name == 'CAS' and change_cas_corr:
        var_name1 += '_corr'
        var_name2 += '_corr'

    nconc_bin1 = data_dict['data'][var_name1][filter_inds]
    nconc_bin2 = data_dict['data'][var_name2][filter_inds]

    naninds = ~np.logical_or(np.isnan(nconc_bin1), np.isnan(nconc_bin2))
    infinds = ~np.logical_or(np.isinf(nconc_bin1), np.isinf(nconc_bin2))

    inds = np.logical_and(naninds, infinds) 

    return nconc_bin1[inds], nconc_bin2[inds] 

def get_overlap_nconc_data(date):

    #print(date)
    adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlr_dict = np.load(adlrfile, allow_pickle=True).item()
    casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    cas_dict = np.load(casfile, allow_pickle=True).item()
    cipfile = DATA_DIR + 'npy_proc/CIP_' + date + '.npy'
    cip_dict = np.load(cipfile, allow_pickle=True).item()

    temp = adlr_dict['data']['temp']
    w = adlr_dict['data']['w']

    filter_inds = np.logical_and.reduce(( \
                            (temp > 27.3), \
                            (w < 1000)))

    cas_var_names = ['nconc_' + str(i) for i in range(12, 17)]
    cip_var_name = 'nconc_1' 

    if change_cas_corr:
        cas_var_names = [var_name + '_corr' for var_name in cas_var_names]

    cas_nconc = np.zeros(np.shape(cas_dict['data']['time'][filter_inds]))
    cip_nconc = cip_dict['data'][cip_var_name][filter_inds]

    for var_name in cas_var_names:
        cas_nconc += cas_dict['data'][var_name][filter_inds]

    naninds = ~np.logical_or(np.isnan(cas_nconc), np.isnan(cip_nconc))
    infinds = ~np.logical_or(np.isinf(cas_nconc), np.isinf(cip_nconc))

    inds = np.logical_and(naninds, infinds) 

    return cas_nconc[inds], cip_nconc[inds] 

def make_and_save_nconc_scatter(nconc_bin1, nconc_bin2, i, instr_name):

    nconc_bin1 = nconc_bin1*1.e-6
    nconc_bin2 = nconc_bin2*1.e-6
    diff = nconc_bin1 - nconc_bin2

    m, b, R, sig = linregress(nconc_bin1, nconc_bin2)
    return R**2

    fig, ax = plt.subplots()

    ax.scatter(nconc_bin1, nconc_bin2)
    ax.set_xlabel('nconc from ' + instr_name + ' bin ' + str(i))
    ax.set_ylabel('nconc from ' + instr_name + ' bin ' + str(i+1))

    regline = b + m*nconc_bin1
    conf_band = get_conf_band(nconc_bin1, nconc_bin2, regline) 
    print(np.sum(np.isinf(nconc_bin1)))
    print(regline)
    print(conf_band)
    ax.scatter(nconc_bin1, regline + conf_band, c='b', alpha=0.4)
    ax.scatter(nconc_bin1, regline - conf_band, c='b', alpha=0.4)
    #ax.fill_between(nconc_bin1, regline + conf_band, \
    #                regline - conf_band, color='b', \
    #                alpha=0.4, label='95% confidence band')

    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c='black', \
            linestyle='dashed', \
            linewidth=2, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
    #ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle=':', c='r')

    #ax.set_xlim([0, 55])
    #ax.set_ylim([0, 55])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()

    outfile = FIG_DIR + 'bin_correlations_' + instr_name + '_' \
                                        + str(i) + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

    return R**2.
    
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

def make_rsq_chart(rsq_vals, x_labels):

    fig, ax = plt.subplots()
    
    print(x_labels)
    print(rsq_vals)
    ax.bar(x_labels, rsq_vals, color=magma_pink)
    #ax.bar(x_labels[2:], anomalous_spectrum_counts[2:], color=magma_pink)
    ax.set_ylabel(r'$R^2$')
    ax.set_xticklabels(x_labels, rotation=90)
    #ax.set_xticklabels(x_labels[2:], rotation=90)
    #ax.get_xticklabels()[10].set_color('red')
    ax.get_xticklabels()[11].set_color('red')
    ax.get_xticklabels()[12].set_color('red')

    outfile = FIG_DIR + 'bin_correlations_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
