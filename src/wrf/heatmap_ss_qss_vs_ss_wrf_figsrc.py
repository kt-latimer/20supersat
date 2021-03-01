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
    
    filename = 'filtered_data_dict.npy'
    data_dict = np.open(DATA_DIR + filename, allow_pickle=True).item()

    for case_label in case_label_dict.keys():
        make_and_save_ss_qss_vs_ss_wrf(data_dict, case_label)

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

    h = ax.hist2d(ss_qss, ss_wrf, bins=ss_bins, cmin=1./(N_points*d_ss**2.), \
        density=True, norm=matplotlib.colors.LogNorm(), cmap=plt.cm.magma_r)
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
