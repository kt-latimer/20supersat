"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np
import sys

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}
                            
lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

###
### physical constants
###
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_e = 6.3781e6 #radius of Earth (m)
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

def main():
    
    for case_label in ['Unpolluted']:#case_label_dict.keys():
        if len(sys.argv) == 2:
            imax = int(sys.argv[1])
        else:
            print('wrong number of arguments')
            return

        make_and_save_imax_heatmap(case_label, case_label_dict[case_label], \
                            cutoff_bins, full_ss, incl_rain, incl_vent, imax)

def make_and_save_imax_heatmap(case_label, case_dir_name, \
            cutoff_bins, full_ss, incl_rain, incl_vent, imax):

    #get lwc file variables 
    lwc_file = Dataset(DATA_DIR + case_dir_name + \
                'wrfout_d01_lwc_spectrum_vars', 'r')
    lwc_vars = lwc_file.variables

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                        'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                    'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_qss = get_ss_qss(met_vars, dsdsum_vars, cutoff_bins, \
                        full_ss, incl_rain, incl_vent)
    ss_wrf = met_vars['ss_wrf'][...]*100

    lwc_var_name = 'r3n_sum_' + str(imax)
    r3n = lwc_vars[lwc_var_name][...]
    lwc = 4./3.*np.pi*rho_w*r3n/rho_air

    #close files for memory
    lwc_file.close()
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    filter_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (temp > 273)))

    ss_qss = ss_qss[filter_inds]
    ss_wrf = ss_wrf[filter_inds]

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
    ax.legend(loc='upper left')

    rmax = str(bin_radii[imax]*1.e6)
    ax.text(0.7, 0.9, r'$r_{max}='+rmax+'\mu m$')
    fig.suptitle('Actual versus approximated supersaturation - WRF ' + case_label)

    outfile = FIG_DIR + 'imax_' + str(imax) + '_heatmap_ss_qss_vs_ss_wrf_' \
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
