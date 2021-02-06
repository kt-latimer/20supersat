"""
1) make heatmap showing least squares slope and r-squared of ss_wrf vs ss_qss
for different vals of w and LWC cutoffs; 2) save ss_wrf and ss_qss
distributions (histograms) in the same parameter space
"""
from itertools import product
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = '/home/klatimer/proj/20supersat/'
DATA_DIR = BASE_DIR + 'data/revmywrf/'
FIG_DIR = BASE_DIR + 'figures/revmywrf/'
from revmywrf.ss_qss_calculations import get_lwc, get_nconc, get_ss, linregress

#for plotting
figversionstr = 'v1_'
distdataversionstr = 'v7_'
lhdataversionstr = 'v2_'
#matplotlib.rcParams.update({'font.size': 40})
matplotlib.rcParams.update({'font.family': 'serif'})
magma = cm.get_cmap('magma')
rev_magma = cm.get_cmap('magma_r')
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
subscript_dict = {'Polluted':'poll', 'Unpolluted':'unpoll'}

log_lwc_min = -6
log_lwc_max = -2
n_lwc_vals = 9
lwc_filter_vals = 10**np.linspace(log_lwc_min, log_lwc_max, n_lwc_vals)

w_min = 0
w_max = 6 
n_w_vals = 7
w_filter_vals = np.linspace(w_min, w_max, n_w_vals)

#modified color structure for histograms
lwc_inds_subset = [4, 5]
w_inds_subset = [1, 2]
magma_discrete = cm.get_cmap('magma', len(lwc_inds_subset)*len(w_inds_subset))
colors = magma_discrete.colors 

d_ss = 0.25
dec_prec = 2

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

data_dict = {'Polluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}, 
             'Unpolluted': {'m_arr': None, 'npts_arr': None, 'rsq_arr': None, \
                            'ss_qss_distbs': None, 'ss_wrf_distbs': None}}

def main():
    
    filename = distdataversionstr + 'systematic_filtering_evaluation_data.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    dist_arr = get_dist_arr(data_dict)

    filename = lhdataversionstr + \
        'lh_frac_only_systematic_filtering_evaluation_data.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    for case_label in ['Unpolluted']: #kinda lazy but turns out unpolluted
                                      #region with lh > 0.5 is a subset of
                                      #polluted region of the same
        lh_filt_arr, lh_tot_arr = \
            get_lh_data(case_label, data_dict)

    make_and_save_combo_heatmap(dist_arr, lh_filt_arr, lh_tot_arr, case_label)

def get_dist_arr(data_dict):

    dist_arr = np.zeros(np.shape(data_dict['Polluted']['m_arr']))

    for key1 in ["Polluted", "Unpolluted"]:
        for key2 in ["m_arr", "rsq_arr"]:
            dist_arr += (1 - data_dict[key1][key2])**2.

    return np.sqrt(dist_arr)
        
def get_lh_data(case_label, data_dict):

    case_dict = data_dict[case_label]

    lh_filt_arr = case_dict['lh_filt_arr']
    lh_tot_arr = case_dict['lh_tot_arr']

    return lh_filt_arr, lh_tot_arr 

def make_and_save_combo_heatmap(dist_arr, lh_filt_arr, lh_tot_arr, case_label):
    """
    adapted from `Creating annotated heatmaps' matplotlib tutorial
    """

    #matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()

    im = ax.imshow(dist_arr.T, cmap=rev_magma)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'$\sqrt{(1-m_{unpoll})^2 + (1-R_{unpoll}^2)^2 + ' \
                            + '(1-m_{poll})^2 + (1-R_{poll}^2)^2}}}}}}}}}$')
    annotate_heatmap(im)

    resolution = 75 

    lh_frac_arr = lh_filt_arr/lh_tot_arr
    f = lambda x,y: lh_frac_arr[int(y),int(x)]
    g = np.vectorize(f)

    x = np.linspace(0,lh_frac_arr.shape[1], lh_frac_arr.shape[1]*resolution)
    y = np.linspace(0,lh_frac_arr.shape[0], lh_frac_arr.shape[0]*resolution)
    X2, Y2= np.meshgrid(x[:-1],y[:-1])
    Z2 = g(X2,Y2)

    plt.contour(Y2-0.5,X2-0.5,Z2, [0.5], colors='w', linewidths=[10])

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    fig.suptitle('Validity of QSS approximation vs data \n filtering scheme - Combined', x=0.6, y=1)

    outfile = FIG_DIR + figversionstr + 'FINAL_from_data_combo_heatmap_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

def annotate_heatmap(im):

    data = im.get_array()
    threshold = im.norm(data.max())/2.
    valfmt="{x:.2f}"
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    textcolors=["black", "white"]
    kw = dict(horizontalalignment="center", \
              verticalalignment="center")

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

if __name__ == "__main__":
    main()
