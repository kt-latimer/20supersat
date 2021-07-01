"""
Bar chart showing equal-time correlation in number concentrations measured by
adjacent bins
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_lo_bin, CAS_mid_bin, CAS_up_bin, \
                                                    CIP_lo_bin, CIP_up_bin
from halo.utils import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]

change_CAS_corr = True
cutoff_bins = True
incl_rain = True 
incl_vent = True
full_ss = True

def main():

    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    rsq_vals = []
    x_labels = []

    for i in range(CAS_lo_bin, CAS_up_bin):
        rsq = make_adjacent_bin_plot(i, 'CAS', CAS_dict)
        rsq_vals.append(rsq)
        x_labels.append('CAS bin ' + str(i) + ' vs ' + str(i+1))
        
    rsq = make_overlap_bin_plot(CAS_dict, CIP_dict)
    rsq_vals.append(rsq)
    x_labels.append('CAS bins 12-16 vs CIP bin 1')

    for i in range(CIP_lo_bin, CIP_up_bin):
        rsq = make_adjacent_bin_plot(i, 'CIP', CIP_dict)
        rsq_vals.append(rsq)
        x_labels.append('CIP bin ' + str(i) + ' vs ' + str(i+1))

    make_rsq_chart(rsq_vals, x_labels)

def make_adjacent_bin_plot(i, instr_name, instr_dict):

    var_name1 = 'nconc_' + str(i)
    var_name2 = 'nconc_' + str(i+1)

    if instr_name == 'CAS' and change_CAS_corr:
        var_name1 += '_corr'
        var_name2 += '_corr'

    nconc_bin1 = instr_dict['data'][var_name1]
    nconc_bin2 = instr_dict['data'][var_name2]

    naninds = ~np.logical_or(np.isnan(nconc_bin1), np.isnan(nconc_bin2))
    infinds = ~np.logical_or(np.isinf(nconc_bin1), np.isinf(nconc_bin2))

    inds = np.logical_and(naninds, infinds) 

    nconc_bin1 = nconc_bin1[inds]
    nconc_bin2 = nconc_bin2[inds] 

    rsq = make_and_save_nconc_scatter(nconc_bin1, nconc_bin2, i, instr_name)

    return rsq 

def make_overlap_bin_plot(CAS_dict, CIP_dict):
    
    CAS_nconc, CIP_nconc = get_overlap_nconc_data(CAS_dict, CIP_dict)
    rsq = make_and_save_nconc_scatter(CAS_nconc, CIP_nconc, 543, 'both')

    return rsq 

def get_overlap_nconc_data(CAS_dict, CIP_dict):

    #cas bins 25-50um diam
    CAS_var_names = ['nconc_' + str(i) for i in \
                range(CAS_mid_bin, CAS_up_bin+1)]
    #cip bin 25-75um diam
    CIP_var_name = 'nconc_1' 

    if change_CAS_corr:
        CAS_var_names = [var_name + '_corr' for var_name in CAS_var_names]

    CAS_nconc = np.zeros(np.shape(CAS_dict['data']['time'][filter_inds]))
    CIP_nconc = CIP_dict['data'][CIP_var_name][filter_inds]

    for var_name in CAS_var_names:
        CAS_nconc += CAS_dict['data'][var_name][filter_inds]

    naninds = ~np.logical_or(np.isnan(CAS_nconc), np.isnan(CIP_nconc))
    infinds = ~np.logical_or(np.isinf(CAS_nconc), np.isinf(CIP_nconc))

    inds = np.logical_and(naninds, infinds) 

    return CAS_nconc[inds], CIP_nconc[inds] 

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
    ax.scatter(nconc_bin1, regline + conf_band, c='b', alpha=0.4)
    ax.scatter(nconc_bin1, regline - conf_band, c='b', alpha=0.4)

    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c='black', \
            linestyle='dashed', \
            linewidth=2, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.legend()

    outfile = FIG_DIR + 'FINAL_bin_correlations_' + instr_name + '_' \
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
    ax.set_ylabel(r'$R^2$')
    ax.set_xticklabels(x_labels, rotation=90)
    #highlight correlations with CIP bin 1
    ax.get_xticklabels()[11].set_color('red')
    ax.get_xticklabels()[12].set_color('red')

    outfile = FIG_DIR + 'FINAL_bin_correlations_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
