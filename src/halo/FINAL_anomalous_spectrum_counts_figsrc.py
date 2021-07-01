"""
Histogram to show frequency of each bin having an anomalously high drop count
relative to other bins at a given point in time (arbitrary definition: higher
than next highest bin by ``cutoff_factor``)
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins, \
        CAS_lo_bin, CAS_up_bin, CIP_lo_bin, CIP_up_bin
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

CAS_n_bins = np.shape(CAS_bins['upper'])[0]
CIP_n_bins = np.shape(CIP_bins['upper'])[0]
CAS_dlogDp = np.log10(CAS_bins['upper']/CAS_bins['lower'])
CIP_dlogDp = np.log10(CIP_bins['upper']/CIP_bins['lower'])
HALO_dlogDp = np.concatenate((CAS_dlogDp, CIP_dlogDp))

cutoff_factor=5

def main():

    anomalous_spectrum_counts = get_anomalous_spectrum_counts()
    make_anomalous_spectrum_count_fig(anomalous_spectrum_counts)

def get_anomalous_spectrum_counts(date):

    CAS_file = DATA_DIR + 'npy_proc/CAS_alldates.npy'
    CAS_dict = np.load(CAS_file, allow_pickle=True).item()
    CIP_file = DATA_DIR + 'npy_proc/CIP_alldates.npy'
    CIP_dict = np.load(CIP_file, allow_pickle=True).item()

    CAS_t = CAS_dict['data']['time']
    anomalous_spectrum_counts = np.zeros(CAS_n_bins + CIP_n_bins) 

    for i, t in enumerate(CAS_t):
        spectrum = []
        for j in range(CAS_lo_bin, CAS_up_bin+1):
            var_name = 'nconc_' + str(j)
            if change_CAS_corr:
                var_name += '_corr'
            nconc_j = CAS_dict['data'][var_name][i]/dlogDp[j-5]
            spectrum.append(nconc_j)
        for j in range(CIP_lo_bin, CIP_up_bin+1):
            var_name = 'nconc_' + str(j)
            nconc_j = CIP_dict['data'][var_name][i]/dlogDp[j-1] 
            spectrum.append(nconc_j)

        i_max, val_max = get_spectrum_max(spectrum)
        spectrum.pop(i_max)
        i_pen_max, val_pen_max = get_spectrum_max(spectrum)
        
        if val_max > cutoff_factor*val_pen_max:
            anomalous_spectrum_counts[i_max] = \
                    anomalous_spectrum_counts[i_max] + 1

    return anomalous_spectrum_counts
        
def get_spectrum_max(spectrum):

    i_max = 0
    val_max = -1

    for i, val in enumerate(spectrum):
        if val > val_max:
            val_max = val
            i_max = i

    return i_max, val_max
        
def make_anomalous_spectrum_count_fig(anomalous_spectrum_counts):

    print(anomalous_spectrum_counts)
    print(len(anomalous_spectrum_counts))
    x_labels = []

    for j in range(CAS_lo_bin, CAS_up_bin+1):
        x_labels.append('CAS bin ' + str(j))
    for j in range(CIP_lo_bin, CIP_up_bin+1):
        x_labels.append('CIP bin ' + str(j))

    fig, ax = plt.subplots()
    
    #remove bins with diam less than 3um
    ax.bar(x_labels[2:], anomalous_spectrum_counts[2:], color=magma_pink)
    ax.set_ylabel('Anomoalous peak count')
    #remove bins with diam less than 3um
    ax.set_xticklabels(x_labels[2:], rotation=90)
    #highlight CIP bin 1
    ax.get_xticklabels()[10].set_color('red')

    outfile = FIG_DIR + 'FINAL_anomalous_spectrum_counts_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
