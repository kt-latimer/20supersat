"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
                            
case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

def main():
    
    filename = 'filtered_data_dict_v3.npy'
    data_dict = np.load(DATA_DIR + filename, allow_pickle=True).item()

    ss_dict = {'Polluted': None, 'Unpolluted': None}

    for case_label in case_label_dict.keys():
        ss_qss, ss_wrf = make_and_save_ss_qss_vs_ss_wrf(data_dict, case_label)
        ss_dict[case_label] = {'ss_qss': ss_qss, 'ss_wrf': ss_wrf}

    ss_qss_combined = np.concatenate((ss_dict['Polluted']['ss_qss'], \
                                        ss_dict['Unpolluted']['ss_qss']))
    ss_wrf_combined = np.concatenate((ss_dict['Polluted']['ss_wrf'], \
                                        ss_dict['Unpolluted']['ss_wrf']))

    m, b, R, sig = linregress(ss_qss_combined, ss_wrf_combined)
    print('combined')
    print(m, b, R**2)

def make_and_save_ss_qss_vs_ss_wrf(data_dict, case_label):

    case_data_dict = data_dict[case_label]

    ss_qss = case_data_dict['ss_qss'] 
    ss_wrf = case_data_dict['ss_wrf'] 

    m, b, R, sig = linregress(ss_qss, ss_wrf)
    print(case_label)
    print(m, b, R**2)
    N_points = np.sum(ss_qss < 200)
    print('# pts total: ' + str(N_points))
    print('max: ' + str(np.nanmax(ss_qss)))
    print('# pts ss > 2%: ' + str(np.sum(ss_qss > 2)))
   
    print_point_count_per_quadrant(ss_qss, ss_wrf)

    fig, ax = plt.subplots()
    
    ss_bins = get_ss_bins(ss_min, ss_max, d_ss)
    rain_rates = get_rain_rates(rain_rate, ss_bins, ss_qss, ss_wrf)

    h = ax.hist2d(ss_bins[:-1], ss_bins[:-1], bins=ss_bins, \
            weights = rain_rates, cmin=np.min(rain_rates), \
            cmap=plt.cm.magma_r)
                #cmin=np.min(rain_rates), density=True, \
                #norm=matplotlib.colors.LogNorm(), cmap=plt.cm.magma_r)
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label(r'$\frac{d^2n_{points}}{dSS_{QSS}dSS_{WRF}}$')

    ax.set_xlim((ss_min, ss_max))
    ax.set_ylim((ss_min, ss_max))
    ax.set_aspect('equal', 'box')

    ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
            c=colors['line'], \
            linestyle='dashed', \
            linewidth=2, \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))

    ax.set_xlabel(r'$SS_{QSS}$ (%)')
    ax.set_ylabel(r'$SS_{WRF}$ (%)')
    plt.legend(loc=2)
    fig.suptitle('Actual versus approximated supersaturation - WRF ' + case_label)

    outfile = FIG_DIR + 'heatmap_ss_qss_vs_ss_wrf_' \
                            + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

    return ss_qss, ss_wrf

def get_ss_bins(ss_min, ss_max, d_ss):

    ss_bins = np.arange(ss_min, ss_max, d_ss)

    return ss_bins

def print_point_count_per_quadrant(ss_qss, ss_wrf):

    n_q1 = np.sum(np.logical_and(ss_qss > 0, ss_wrf > 0))
    n_q2 = np.sum(np.logical_and(ss_qss < 0, ss_wrf > 0))
    n_q3 = np.sum(np.logical_and(ss_qss < 0, ss_wrf < 0))
    n_q4 = np.sum(np.logical_and(ss_qss > 0, ss_wrf < 0))
    
    print('Number of points in Q1:', n_q1)
    print('Number of points in Q2:', n_q2)
    print('Number of points in Q3:', n_q3)
    print('Number of points in Q4:', n_q4)
    print()
    print('Domain:', np.min(ss_qss), np.max(ss_qss))
    print('Range:', np.min(ss_wrf), np.max(ss_wrf))

if __name__ == "__main__":
    main()
