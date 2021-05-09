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

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2. 

def main():
    
    for case_label in case_label_dict.keys():
        dsd_dict, vent_dsd_dict = get_dsd_dicts(case_label)
        make_and_save_avg_dsd(case_label, dsd_dict, vent_dsd_dict)

    ddpoll, vddpoll = get_dsd_dicts('Polluted')
    ddunpoll, vddunpoll = get_dsd_dicts('Unpolluted')
    r1 = ddpoll['mean']/ddunpoll['mean']
    r2 = vddpoll['mean']/vddunpoll['mean']
    for i, thing in enumerate(r1):
        print(thing, r2[i], bin_radii[i])

def get_dsd_dicts(case_label):

    print(case_label)

    case_dsd_filename = DATA_DIR + 'dsd_dict_' + case_label + '_data.npy'
    case_vent_dsd_filename = DATA_DIR + 'vent_dsd_dict_' \
                                + case_label + '_data.npy'

    dsd_dict = np.load(case_dsd_filename, allow_pickle=True).item()
    vent_dsd_dict = np.load(case_vent_dsd_filename, allow_pickle=True).item()

    return dsd_dict, vent_dsd_dict

def make_and_save_avg_dsd(case_label, dsd_dict, vent_dsd_dict):

    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)

    ax11.errorbar(bin_radii*1.e6, dsd_dict['mean']*1.e-6, \
                        yerr = dsd_dict['std']*1.e-6, fmt='o')
    ax12.errorbar(bin_radii*1.e6, dsd_dict['mean']*bin_radii, \
                        yerr = dsd_dict['std']*bin_radii, fmt='o')
    ax21.errorbar(bin_radii*1.e6, dsd_dict['median']*1.e-6, \
                        yerr = [dsd_dict['lo_quart']*1.e-6, \
                        dsd_dict['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(bin_radii*1.e6, dsd_dict['median']*bin_radii, \
                        yerr = [dsd_dict['lo_quart']*bin_radii, \
                        dsd_dict['up_quart']*bin_radii], fmt='o')

    ax11.set_xscale('log')
    ax12.set_xscale('log')
    ax21.set_xscale('log')
    ax22.set_xscale('log')
    ax11.set_yscale('log')
    ax12.set_yscale('log')
    ax21.set_yscale('log')
    ax22.set_yscale('log')

    outfile = FIG_DIR + 'avg_dsd_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

    fig, [[ax11, ax12], [ax21, ax22]] = plt.subplots(2, 2)

    ax11.errorbar(bin_radii*1.e6, vent_dsd_dict['mean']*1.e-6, \
                        yerr = vent_dsd_dict['std'], fmt='o')
    ax12.errorbar(bin_radii*1.e6, vent_dsd_dict['mean']*bin_radii, \
                        yerr = vent_dsd_dict['std']*bin_radii, fmt='o')
    ax21.errorbar(bin_radii*1.e6, vent_dsd_dict['median']*1.e-6, \
                        yerr = [vent_dsd_dict['lo_quart']*1.e-6, \
                        vent_dsd_dict['up_quart']*1.e-6], fmt='o')
    ax22.errorbar(bin_radii*1.e6, vent_dsd_dict['median']*bin_radii, \
                        yerr = [vent_dsd_dict['lo_quart']*bin_radii, \
                        vent_dsd_dict['up_quart']*bin_radii], fmt='o')

    ax11.set_xscale('log')
    ax12.set_xscale('log')
    ax21.set_xscale('log')
    ax22.set_xscale('log')
    ax11.set_yscale('log')
    ax12.set_yscale('log')
    ax21.set_yscale('log')
    ax22.set_yscale('log')

    outfile = FIG_DIR + 'avg_vent_dsd_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

if __name__ == "__main__":
    main()
