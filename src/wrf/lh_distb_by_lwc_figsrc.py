"""
make smaller data files for quantities after applying LWC, vertical wind
velocity, and temperature filters
"""
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

from wrf import BASE_DIR, DATA_DIR, FIG_DIR
from wrf.ss_functions import get_lwc, get_nconc, get_ss_qss, linregress

#for plotting
matplotlib.rcParams.update({'font.family': 'serif'})

case_label_dict = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}

def main():
    
    for case_label in case_label_dict.keys():
        make_and_save_lh_cdf(case_label) 

def make_and_save_lh_cdf(case_label):

    case_dir_name = case_label_dict[case_label]

    #get met file variables 
    met_file = Dataset(DATA_DIR + case_dir_name + \
                                'wrfout_d01_met_vars', 'r')
    met_vars = met_file.variables

    #get relevant physical qtys
    lh = met_vars['LH'][...]
    lwc_cloud = met_vars['LWC_cloud'][...]
    lwc_cloud_and_rain = lwc_cloud + met_vars['LWC_rain'][...]
    temp = met_vars['temp'][...]

    #filtering to determine total condensational latent heating
    filter_inds = np.logical_and.reduce((
                    (lh > 0), \
                    (temp > 273)))

    lh = lh[filter_inds]
    lwc_cloud = lwc_cloud[filter_inds]
    lwc_cloud_and_rain = lwc_cloud_and_rain[filter_inds]

    del filter_inds, temp #for memory

    #close file for memory
    met_file.close()
    
    lwc_cloud_bins = get_lwc_bins(lwc_cloud)
    lh_cdf_cloud = get_lh_cdf(lh, lwc_cloud, lwc_cloud_bins)
    lwc_cloud_and_rain_bins = get_lwc_bins(lwc_cloud_and_rain)
    lh_cdf_cloud_and_rain = get_lh_cdf(lh, \
                lwc_cloud_and_rain, lwc_cloud_and_rain_bins)
    print(case_label)
    print(lwc_cloud_bins)
    print(lh_cdf_cloud)
    print(lwc_cloud_and_rain_bins)
    print(lh_cdf_cloud_and_rain)

    fig, ax = plt.subplots()

    ax.plot(lwc_cloud_bins, lh_cdf_cloud, label=r'LWC$_{cloud}$')
    ax.plot(lwc_cloud_and_rain_bins, lh_cdf_cloud_and_rain, \
                                label=r'LWC$_{cloud and rain}$')
    ax.set_xlabel(r'LWC (g/g)')
    ax.set_ylabel('(Positive) LH CDF')
    ax.legend()

    outfile = FIG_DIR + 'lh_distb_by_lwc_' + case_label + '_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

def get_lwc_bins(lwc):

    lwc_bins = np.linspace(np.min(lwc), np.max(lwc), 40)

    return lwc_bins

def get_lh_cdf(lh, lwc, lwc_bins):

    lh_cdf = [0]
    lh_tot = np.sum(lh)

    for i, low_bin_edge in enumerate(lwc_bins[:-1]):
        hi_bin_edge = lwc_bins[i+1]
        bin_filter_inds = np.logical_and.reduce((
                            (lwc >= low_bin_edge), \
                            (lwc < hi_bin_edge)))
        lh_filt = lh[bin_filter_inds]
        bin_lh_contribution = np.sum(lh_filt) 
        lh_cdf.append(lh_cdf[i] + bin_lh_contribution/lh_tot)

    return np.array(lh_cdf)

if __name__ == "__main__":
    main()
