"""
make and save histograms showing SS_QSS distribution from HALO CAS measurements
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import ticker
from matplotlib.lines import Line2D
import numpy as np

from halo import DATA_DIR, FIG_DIR, CAS_bins, CIP_bins
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

n_cas_bins = np.shape(CAS_bins['upper'])[0]
n_cip_bins = np.shape(CIP_bins['upper'])[0]
cas_dlogDp = np.log10(CAS_bins['upper']/CAS_bins['lower'])
cip_dlogDp = np.log10(CIP_bins['upper']/CIP_bins['lower'])
dlogDp = np.concatenate((cas_dlogDp, cip_dlogDp))

factor=5

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    anomalous_spectrum_counts = np.zeros(n_cas_bins + n_cip_bins) 

    for date in good_dates:
        counts = get_anomalous_spectrum_counts(date)
        anomalous_spectrum_counts += counts 

    make_anomalous_spectrum_count_fig(anomalous_spectrum_counts)

def get_anomalous_spectrum_counts(date):

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

    cas_t = cas_dict['data']['time'][filter_inds]
    
    anomalous_spectrum_counts = np.zeros(n_cas_bins + n_cip_bins) 

    for i, t in enumerate(cas_t):
        spectrum = []
        for j in range(5, 17):
            var_name = 'nconc_' + str(j)
            if change_cas_corr:
                var_name += '_corr'
            nconc_j = cas_dict['data'][var_name][i]/dlogDp[j-5]
            spectrum.append(nconc_j)
        for j in range(1, 20):
            var_name = 'nconc_' + str(j)
            nconc_j = cip_dict['data'][var_name][i]/dlogDp[j-1] 
            spectrum.append(nconc_j)

        i_max, val_max = get_spectrum_max(spectrum)
        spectrum.pop(i_max)
        i_pen_max, val_pen_max = get_spectrum_max(spectrum)
        
        if val_max > factor*val_pen_max:
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

    for j in range(5, 17):
        x_labels.append('CAS bin ' + str(j))
    for j in range(1, 20):
        x_labels.append('CIP bin ' + str(j))

    fig, ax = plt.subplots()
    
    #ax.bar(x_labels, anomalous_spectrum_counts, color=magma_pink)
    ax.bar(x_labels[2:], anomalous_spectrum_counts[2:], color=magma_pink)
    ax.set_ylabel('Anomoalous peak count')
    #ax.set_xticklabels(x_labels, rotation=90)
    ax.set_xticklabels(x_labels[2:], rotation=90)
    ax.get_xticklabels()[10].set_color('red')
    #ax.get_xticklabels()[12].set_color('red')

    outfile = FIG_DIR + 'v2_anomalous_spectrum_counts_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
