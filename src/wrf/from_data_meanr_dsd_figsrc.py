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
upper_bin_radii = 2**(1./6.)*bin_radii
lower_bin_radii = 2**(-1./6.)*bin_radii
log_bin_widths = np.array([np.log10(2.**(1./3.)) for i in range(33)])

def main():
    
    for case_label in case_label_dict.keys():
        dsd_dict, vent_dsd_dict = get_dsd_dicts(case_label)
        make_and_save_meanr_dsd(case_label, dsd_dict, vent_dsd_dict)

    #ddpoll, vddpoll = get_dsd_dicts('Polluted')
    #ddunpoll, vddunpoll = get_dsd_dicts('Unpolluted')
    #r1 = ddpoll['mean']/ddunpoll['mean']
    #r2 = vddpoll['mean']/vddunpoll['mean']
    #for i, thing in enumerate(r1):
    #    print(thing, r2[i], bin_radii[i])

def get_dsd_dicts(case_label):

    print(case_label)

    case_dsd_filename = DATA_DIR + 'dsd_dict_' + case_label + '_data.npy'
    case_vent_dsd_filename = DATA_DIR + 'vent_dsd_dict_' \
                                + case_label + '_data.npy'

    dsd_dict = np.load(case_dsd_filename, allow_pickle=True).item()
    vent_dsd_dict = np.load(case_vent_dsd_filename, allow_pickle=True).item()

    for key in dsd_dict.keys():
        dsd_dict[key] = bin_radii**3.*dsd_dict[key]
        vent_dsd_dict[key] = bin_radii**3.*vent_dsd_dict[key]

    return dsd_dict, vent_dsd_dict

def make_and_save_meanr_dsd(case_label, dsd_dict, vent_dsd_dict):

    fig, ax = plt.subplots()

    ax.errorbar(bin_radii*1.e6, dsd_dict['mean']/log_bin_widths, \
                yerr=dsd_dict['std']/log_bin_widths, color='grey', \
                alpha=0.3, fmt='o')
    ax.step(lower_bin_radii*1.e6, dsd_dict['mean']/log_bin_widths, \
            where='post', color=magma_pink)
    ax.plot([lower_bin_radii[-1]*1.e6, upper_bin_radii[-1]*1.e6], \
            [dsd_dict['mean'][-1]/log_bin_widths[-1], \
            dsd_dict['mean'][-1]/log_bin_widths[-1]], \
            color=magma_pink)
    #ax.errorbar(bin_radii*1.e6, vent_dsd_dict['mean']/log_bin_widths, \
    #            yerr=vent_dsd_dict['std']/log_bin_widths, color='grey', \
    #            alpha=0.3, fmt='o')
    #ax.step(lower_bin_radii*1.e6, vent_dsd_dict['mean']/log_bin_widths, \
    #        where='post', color=magma_pink)
    #ax.plot([lower_bin_radii[-1]*1.e6, upper_bin_radii[-1]*1.e6], \
    #        [vent_dsd_dict['mean'][-1]/log_bin_widths[-1], \
    #        vent_dsd_dict['mean'][-1]/log_bin_widths[-1]], \
    #        color=magma_pink)
    ##ax.errorbar(bin_radii*1.e6, vent_dsd_dict['median']/log_bin_widths, \
    #            yerr=[vent_dsd_dict['lo_quart']/log_bin_widths, \
    #                    vent_dsd_dict['up_quart']/log_bin_widths], \
    #                    color='grey', alpha=0.3, fmt='o')
    #ax.step(lower_bin_radii*1.e6, vent_dsd_dict['median']/log_bin_widths, \
    #        where='post', color=magma_pink)
    #ax.plot([lower_bin_radii[-1]*1.e6, upper_bin_radii[-1]*1.e6], \
    #        [vent_dsd_dict['median'][-1]/log_bin_widths[-1], \
    #        vent_dsd_dict['median'][-1]/log_bin_widths[-1]], \
    #        color=magma_pink)

    ax.set_xlabel(r'r ($\mu$m)')
    ax.set_ylabel(r'$\frac{d(r \cdot f(r) \cdot N(r))}{d\log r}$ (cm$^{-3}$)')

    #ax.set_ylim([1.e-5, 1.e8])
    ax.set_xlim([1.e-1, 1.e4])

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    outfile = FIG_DIR + 'avg_meanr_' + case_label + '_figure.png'
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()    

if __name__ == "__main__":
    main()
