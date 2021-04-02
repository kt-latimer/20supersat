"""
heatmap scatter plot showing agreement bt ss_qss and ss_wrf
don't include contribution from rain drops
don't make ventilation corrections
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from netCDF4 import Dataset, MFDataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.dsd_data_functions import get_bin_nconc
from wrf.ss_functions import get_lwc

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})
colors_arr = cm.get_cmap('magma', 10).colors
magma_pink = colors_arr[5]
                            
lwc_filter_val = 1.e-4
w_cutoff = 1

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

ss_min = -20
d_ss = 0.25
ss_max = 50+d_ss

cutoff_bins = True 
incl_rain = True 
incl_vent = True 
full_ss = True 

bin_diams = np.array([4*(2.**(i/3.))*10**(-6) for i in range(33)]) #bin diams in m
bin_radii = bin_diams/2.
bin_widths = bin_radii*(2.**(1./6.) - 2.**(-1./6.)) #edges are geometric means
                                                    #of center radius values
bin_widths_um = bin_widths*1.e6
log_bin_widths = np.array([np.log10(2.**(1./3.)) for i in range(33)])

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_dsd_q1_vs_q4(case_label, case_label_dict[case_label], \
                                    cutoff_bins, full_ss, incl_rain, incl_vent)

def make_and_save_dsd_q1_vs_q4(case_label, case_dir_name, \
                    cutoff_bins, full_ss, incl_rain, incl_vent):

    print(case_label)

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get dsd sum file variables
    dsdsum_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_all_dsdsum_vars_v2', 'r')
    dsdsum_vars = dsdsum_file.variables

    #get relevant physical qtys
    lwc = met_vars['LWC_cloud'][...] + met_vars['LWC_rain'][...]
    rho_air = met_vars['rho_air'][...]
    temp = met_vars['temp'][...]
    w = met_vars['w'][...]
    ss_wrf = met_vars['ss_wrf'][...]*100

    #close files for memory
    met_file.close()
    dsdsum_file.close()

    #apply filtering criteria
    q1_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (ss_wrf > 0), \
                    (temp > 273)))

    q4_inds = np.logical_and.reduce((
                    (lwc > lwc_filter_val), \
                    (w > w_cutoff), \
                    (ss_wrf < 0), \
                    (temp > 273)))

    del lwc, ss_wrf, temp, w #for memory

    #get raw input file vars (with dsd data)
    input_file = MFDataset(DATA_DIR + case_dir_name + 'wrfout_d01_2014*', 'r')
    input_vars = input_file.variables

    avg_nconc_q1 = np.zeros(np.shape(bin_radii))
    avg_nconc_q4 = np.zeros(np.shape(bin_radii))
    std_nconc_q1 = np.zeros(np.shape(bin_radii))
    std_nconc_q4 = np.zeros(np.shape(bin_radii))

    for i in range(1, 33):
        r_i = bin_radii[i-1]
        nconc_i = get_bin_nconc(i, input_vars, rho_air)
        nconc_i_q1 = nconc_i[q1_inds]
        nconc_i_q4 = nconc_i[q4_inds]
        avg_nconc_q1[i-1] = np.nanmean(nconc_i_q1)
        avg_nconc_q4[i-1] = np.nanmean(nconc_i_q4)
        std_nconc_q1[i-1] = np.nanstd(nconc_i_q1)
        std_nconc_q4[i-1] = np.nanstd(nconc_i_q4)

    #close file for memory
    input_file.close()

    print(avg_nconc_q1)
    print(std_nconc_q1)
    print()
    print(avg_nconc_q4)
    print(std_nconc_q4)
    print()
    
    fig, [ax1, ax2] = plt.subplots(1, 2)
    
    #dNdr_q1 = avg_nconc_q1*1.e-6/bin_widths_um
    #sig_dNdr_q1 = std_nconc_q1*1.e-6/bin_widths_um
    #dNdr_q4 = avg_nconc_q4*1.e-6/bin_widths_um
    #sig_dNdr_q4 = std_nconc_q4*1.e-6/bin_widths_um

    dNdlogr_q1 = avg_nconc_q1*1.e-6/log_bin_widths
    sig_dNdlogr_q1 = std_nconc_q1*1.e-6/log_bin_widths
    dNdlogr_q4 = avg_nconc_q4*1.e-6/log_bin_widths
    sig_dNdlogr_q4 = std_nconc_q4*1.e-6/log_bin_widths

    #ax1.plot(bin_radii*1.e6, dNdr_q1, c=magma_pink)
    #ax1.plot(bin_radii*1.e6, dNdr_q1 + sig_dNdr_q1, c=magma_pink, ls='--')
    #ax1.plot(bin_radii*1.e6, dNdr_q1 - sig_dNdr_q1, c=magma_pink, ls='--')
    #ax1.set_xlabel(r'r ($\mu$m)')
    #ax1.set_ylabel(r'$\frac{dN_{Q1}}{dr}$ ($\frac{1}{cm^3 \mu m}$)')

    #ax2.plot(bin_radii*1.e6, dNdr_q4, c=magma_pink)
    #ax2.plot(bin_radii*1.e6, dNdr_q4 + sig_dNdr_q4, c=magma_pink, ls='--')
    #ax2.plot(bin_radii*1.e6, dNdr_q4 - sig_dNdr_q4, c=magma_pink, ls='--')
    #ax2.set_xlabel(r'r ($\mu$m)')
    #ax2.set_ylabel(r'$\frac{dN_{Q1}}{dr}$ ($\frac{1}{cm^3 \mu m}$)')

    ax1.plot(bin_radii*1.e6, dNdlogr_q1, c=magma_pink)
    ax1.plot(bin_radii*1.e6, dNdlogr_q1 + sig_dNdlogr_q1, c=magma_pink, ls='--')
    ax1.plot(bin_radii*1.e6, dNdlogr_q1 - sig_dNdlogr_q1, c=magma_pink, ls='--')
    ax1.set_xlabel(r'r ($\mu$m)')
    ax1.set_ylabel(r'$\frac{dN_{Q1}}{dlogr}$ ($\frac{1}{cm^3 \mu m}$)')
    ax1.set_xscale('log')

    ax2.plot(bin_radii*1.e6, dNdlogr_q4, c=magma_pink)
    ax2.plot(bin_radii*1.e6, dNdlogr_q4 + sig_dNdlogr_q4, c=magma_pink, ls='--')
    ax2.plot(bin_radii*1.e6, dNdlogr_q4 - sig_dNdlogr_q4, c=magma_pink, ls='--')
    ax2.set_xlabel(r'r ($\mu$m)')
    ax2.set_ylabel(r'$\frac{dN_{Q1}}{dlogr}$ ($\frac{1}{cm^3 \mu m}$)')
    ax2.set_xscale('log')

    outfile = FIG_DIR + 'dsd_q1_vs_q4_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close()    

if __name__ == "__main__":
    main()
