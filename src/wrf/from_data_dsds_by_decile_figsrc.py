"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]
                            
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

n_bins = 33

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(n_bins)]) #bin diams in m
bin_radii = bin_diams/2. 
dlogDp = np.array([2.**(1./3.) for i in range(n_bins)])

def main():
    
    for case_label in case_label_dict.keys():
        spectrum_dict = get_spectrum_dict(case_label)
        make_and_save_dsds_by_decile_graphs(case_label, spectrum_dict)

def get_spectrum_dict(case_label):

    print(case_label)

    spectrum_filename = DATA_DIR + 'dsds_by_decile_' + case_label + '_data.npy'

    spectrum_dict = np.load(spectrum_filename, allow_pickle=True).item()
    for i in range(10):
        decile_key = 'decile_' + str(i+1)
        for bin_ind in range(n_bins):
            spectrum_dict[decile_key]['nconc'][bin_ind] = \
                spectrum_dict[decile_key]['nconc'][bin_ind].data

    return spectrum_dict

def make_and_save_dsds_by_decile_graphs(case_label, spectrum_dict):

    for i in range(9):
        decile_key = 'decile_' + str(i+1)
        fig, ax = plt.subplots()
        #print(decile_key, spectrum_dict[decile_key])
        print(i)
        print(len(spectrum_dict[decile_key]['nconc']))
        print(len(spectrum_dict[decile_key]['vent_coeff']))
        decile_spectra = get_decile_spectra(spectrum_dict[decile_key])
        for j, spectrum in enumerate(decile_spectra):
            ax.plot(1.e6*bin_radii, spectrum, c=colors_arr[i])

        ax.set_xscale('log')
        ax.set_yscale('log')
        outfile = FIG_DIR + 'dsds_by_decile_' + str(i+1) + '_' \
                                    + case_label + '_figure.png'
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()    

def get_decile_spectra(decile_dsd_dict):

    dsd = np.array(decile_dsd_dict['vent_coeff'])* \
                    np.array(decile_dsd_dict['nconc'])
    print(dsd.shape)
    dsd = np.transpose(dsd)
    #print(np.shape(dsd))
    decile_spectra = []
    for i, row in enumerate(dsd):
        #print(i, row.shape)
        dfNrdlogDp = row*bin_radii/dlogDp
        decile_spectra.append(dfNrdlogDp)

    return decile_spectra 

if __name__ == "__main__":
    main()
