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

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
rev_magma = cm.get_cmap('magma_r')
                            
d_ss = 0.25
dec_prec = 2

def main():
    
    heatmap_filename = 'filtering_criteria_data.npy'
    heatmap_data_dict = np.load(DATA_DIR + heatmap_filename, allow_pickle=True).item()

    main_filename = 'filtered_data_dict.npy'
    main_data_dict = np.load(DATA_DIR + main_filename, allow_pickle=True).item()

    print(heatmap_data_dict['Polluted']['m_arr'])
    print(heatmap_data_dict['Polluted']['rsq_arr'])
    print(heatmap_data_dict['Unpolluted']['m_arr'])
    print(heatmap_data_dict['Unpolluted']['rsq_arr'])
    dist_arr = np.sqrt((1-heatmap_data_dict['Polluted']['m_arr'])**2. + \
                        (1-heatmap_data_dict['Polluted']['rsq_arr'])**2. + \
                        (1-heatmap_data_dict['Unpolluted']['m_arr'])**2. + \
                        (1-heatmap_data_dict['Unpolluted']['rsq_arr'])**2.)
    
    print(dist_arr)
    
    lh_from_filtered_poll = heatmap_data_dict['Polluted']['lh_sum_arr']
    lh_tot_poll = main_data_dict['Polluted']['lh_tot']
    lh_frac_poll = lh_from_filtered_poll/lh_tot_poll

    lh_from_filtered_unpoll = heatmap_data_dict['Unpolluted']['lh_sum_arr']
    lh_tot_unpoll = main_data_dict['Unpolluted']['lh_tot']
    lh_frac_unpoll = lh_from_filtered_unpoll/lh_tot_unpoll

    lh_frac_arr = np.minimum(lh_frac_poll, lh_frac_unpoll)
    print(lh_frac_arr)

    lwc_filter_vals = heatmap_data_dict['Polluted']['lwc_cutoffs_arr']
    w_filter_vals = heatmap_data_dict['Polluted']['w_cutoffs_arr']
    print(lwc_filter_vals)
    print(w_filter_vals)
    
    make_and_save_filtering_criteria_heatmap(dist_arr, lh_frac_arr, \
                                    lwc_filter_vals, w_filter_vals)

def make_and_save_filtering_criteria_heatmap(dist_arr, lh_frac_arr, \
                                    lwc_filter_vals, w_filter_vals):
    """
    adapted from `Creating annotated heatmaps' matplotlib tutorial
    """

    fig, ax = plt.subplots()

    im = ax.imshow(dist_arr.T, cmap=rev_magma)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'$\sqrt{(1-m_{unpoll})^2 + (1-R_{unpoll}^2)^2 + ' \
                            + '(1-m_{poll})^2 + (1-R_{poll}^2)^2}}}}}}}}}$')
    annotate_heatmap(im)

    resolution = 75 

    f = lambda x,y: lh_frac_arr[int(y),int(x)]
    g = np.vectorize(f)

    x = np.linspace(0,lh_frac_arr.shape[1], lh_frac_arr.shape[1]*resolution)
    y = np.linspace(0,lh_frac_arr.shape[0], lh_frac_arr.shape[0]*resolution)
    X2, Y2= np.meshgrid(x[:-1],y[:-1])
    Z2 = g(X2,Y2)

    plt.contour(Y2-0.5,X2-0.5,Z2, [0.5], colors='w', linewidths=[10])

    n_lwc_vals = np.shape(lwc_filter_vals)[0]
    n_w_vals = np.shape(w_filter_vals)[1]

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels(np.around(np.log10(lwc_filter_vals), 2))
    ax.set_yticklabels(np.around(w_filter_vals, 2))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    fig.suptitle('Validity of QSS approximation vs data \n filtering scheme - Combined', x=0.6, y=1)

    outfile = FIG_DIR + 'filtering_criteria_figure.png'
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
