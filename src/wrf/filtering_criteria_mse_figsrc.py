"""
"""
from itertools import product
import matplotlib
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
rev_magma = cm.get_cmap('magma_r')
                            
def main():
    
    heatmap_filename = 'FINAL_filtering_criteria_data.npy'
    heatmap_data_dict = np.load(DATA_DIR + heatmap_filename, allow_pickle=True).item()

    main_filename = 'filtered_data_dict.npy'
    main_data_dict = np.load(DATA_DIR + main_filename, allow_pickle=True).item()

    mse_poll = heatmap_data_dict['Polluted']['mse_arr']
    mse_unpoll = heatmap_data_dict['Unpolluted']['mse_arr']
    npts_poll = heatmap_data_dict['Polluted']['npts_arr']
    npts_unpoll = heatmap_data_dict['Unpolluted']['npts_arr']
    dist_arr = (mse_poll*npts_poll + mse_unpoll*npts_unpoll)/ \
                (npts_poll + npts_unpoll)
    
    lh_from_filtered_poll = heatmap_data_dict['Polluted']['lh_sum_arr']
    lh_tot_poll = main_data_dict['Polluted']['lh_tot']
    lh_frac_poll = lh_from_filtered_poll/lh_tot_poll

    lh_from_filtered_unpoll = heatmap_data_dict['Unpolluted']['lh_sum_arr']
    lh_tot_unpoll = main_data_dict['Unpolluted']['lh_tot']
    lh_frac_unpoll = lh_from_filtered_unpoll/lh_tot_unpoll

    lh_frac_arr = np.minimum(lh_frac_poll, lh_frac_unpoll)

    lwc_filter_vals = heatmap_data_dict['Polluted']['lwc_cutoffs_arr']
    w_filter_vals = heatmap_data_dict['Polluted']['w_cutoffs_arr']
    
    make_and_save_filtering_criteria_heatmap(dist_arr, lh_frac_arr, \
                                    lwc_filter_vals, w_filter_vals)

def make_and_save_filtering_criteria_heatmap(dist_arr, lh_frac_arr, \
                                    lwc_filter_vals, w_filter_vals):
    """
    adapted from `Creating annotated heatmaps' matplotlib tutorial
    """

    fig, ax = plt.subplots()

    im = ax.imshow(np.flip(dist_arr.T, axis=0), cmap=rev_magma)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('MSE')
    annotate_heatmap(im)
    add_hatching(ax, lh_frac_arr)

    resolution = 75 

    f = lambda x,y: lh_frac_arr[int(y),int(x)]
    g = np.vectorize(f)

    x = np.linspace(0,lh_frac_arr.shape[1], lh_frac_arr.shape[1]*resolution)
    y = np.linspace(0,lh_frac_arr.shape[0], lh_frac_arr.shape[0]*resolution)
    X2, Y2= np.meshgrid(x[:-1],y[:-1])
    Z2 = g(X2,Y2)

    n_lwc_vals = np.shape(lwc_filter_vals)[0]
    n_w_vals = np.shape(w_filter_vals)[1]

    ax.set_xticks(np.arange(n_lwc_vals))
    ax.set_yticks(np.arange(n_w_vals))
    ax.set_xticklabels([str(lwc) for lwc in \
        np.around(np.log10(lwc_filter_vals[:,0]), 2)])
    ax.set_yticklabels(np.flip(np.array([str(w) for w in \
                        np.around(w_filter_vals[0], 2)])))
    ax.set_xbound([-0.5, 6.5])
    ax.set_ybound([-0.5, -0.5 + n_w_vals])

    ax.set_xlabel('Min log(LWC) cutoff (kg/kg)')
    ax.set_ylabel('Min w cutoff (m/s)')

    fig.suptitle('MSE - QSS approximation vs true SS', x=0.6, y=1)

    outfile = FIG_DIR + 'FINAL_filtering_criteria_mse_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

def annotate_heatmap(im):

    data = im.get_array()
    valfmt="{x:.2f}"
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    kw = dict(horizontalalignment="center", \
              verticalalignment="center", 
              bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            kw.update(color='black')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

def add_hatching(ax, lh_frac_arr):

    for i, row in enumerate(lh_frac_arr):
        for j, val in enumerate(row):
            if val < 0.5: 
                rect = patches.Rectangle((i-0.5, 5-(j-0.5)), 1, 1, linewidth=0, \
                        hatch='//', edgecolor='gray', facecolor='none')
                ax.add_patch(rect)

if __name__ == "__main__":
    main()
