"""
make and save histograms showing aerosol nconc distribution from HALO PCASP measurements
"""
import csv
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revhalo import DATA_DIR, FIG_DIR, PCASP_bins, UHSAS_bins, UHSAS2_bins
from revhalo.revhalo_data_polish import match_two_arrays
from revhalo.ss_qss_calculations import get_lwc_from_cas

#for plotting
versionstr = 'v2_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'polluted': '#87cfeb', 'unpolluted': '#0b0bf7'}
mplcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', \
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

lwc_filter_val = 1.e-5 #v1-3
#lwc_filter_val = 10 #v4
n_samples = -1 #v1, v4
#n_samples = 60 #v2 
#n_samples = 1 #v3

change_cas_corr = True 
cutoff_bins = True 
incl_rain = True
incl_vent = True

#diams and differentials
bin_diams = {'PCASP': (PCASP_bins['upper'] + PCASP_bins['lower'])/2., \
         'UHSAS': (UHSAS_bins['upper'] + UHSAS_bins['lower'])/2., \
         'UHSAS2': (UHSAS2_bins['upper'] + UHSAS2_bins['lower'])/2.}
bin_diam_diffls = {'PCASP': (PCASP_bins['upper'] - PCASP_bins['lower']), \
                'UHSAS': (UHSAS_bins['upper'] - UHSAS_bins['lower']), \
                'UHSAS2': (UHSAS2_bins['upper'] - UHSAS2_bins['lower'])}
n_bins = {'PCASP': PCASP_bins['upper'].shape[0], \
          'UHSAS': UHSAS_bins['upper'].shape[0], \
          'UHSAS2': UHSAS2_bins['upper'].shape[0]}
bin_start_inds = {'PCASP': 1, 'UHSAS': 15, 'UHSAS2': 4}

#cpc diams
min_diam_CPC0_resid = 4.e-9
max_diam_CPC0_resid = 0.1e-6 
min_diam_CPC3_resid = 10.e-9
max_diam_CPC3_resid = 0.1e-6 
cpc0_xrange = np.array([min_diam_CPC0_resid, max_diam_CPC0_resid])
cpc3_xrange = np.array([min_diam_CPC3_resid, max_diam_CPC3_resid])
dlogDp_CPC0 = np.log10(max_diam_CPC0_resid/min_diam_CPC0_resid)
dlogDp_CPC3 = np.log10(max_diam_CPC3_resid/min_diam_CPC3_resid)

#fan C_PI cutoff
unpoll_cutoff = 18 

def main():

    fan_aero_size_distb = get_fan_aero_size_distb()
    
    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    uhsas_aero_size_distb_alldates = None
    dN_CPC0_alldates = None
    dN_CPC3_alldates = None
    uhsas_diam_vals_alldates = None

    for date in good_dates:
        if date == '20140906' or date == '20140921': #dates dne for pcasp
            continue

        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlr_dict = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        cas_dict = np.load(casfile, allow_pickle=True).item()
        cpcfile = DATA_DIR + 'npy_proc/CPC_' + date + '.npy'
        cpc_dict = np.load(cpcfile, allow_pickle=True).item()
        uhsasfile = DATA_DIR + 'npy_proc/UHSAS_' + date + '.npy'
        uhsas_dict = np.load(uhsasfile, allow_pickle=True).item()

        uhsas_setname = get_uhsas_setname(date)

        lwc = get_lwc_from_cas(cas_dict, change_cas_corr, cutoff_bins)
        temp = adlr_dict['data']['temp']
        
        filter_inds = np.logical_and.reduce((
                        (lwc < lwc_filter_val), \
                        (temp > 273)))

        print(np.sum(filter_inds))
        print(np.shape(filter_inds))

        #data for this date
        uhsas_aero_size_distb = get_aero_size_distb(uhsas_dict['data'], \
                                            uhsas_setname, filter_inds, n_samples)
        (dN_CPC0, dN_CPC3) = get_residual_nconc(uhsas_dict['data'], \
                                            cpc_dict['data'], filter_inds, \
                                            n_samples, uhsas_setname)
                                   
        #data for all dates combined
        uhsas_aero_size_distb_alldates = add_to_alldates_array(uhsas_aero_size_distb, \
                                    uhsas_aero_size_distb_alldates)
        dN_CPC0_alldates = add_to_alldates_array(dN_CPC0, dN_CPC0_alldates)
        dN_CPC3_alldates = add_to_alldates_array(dN_CPC3, dN_CPC3_alldates)
        uhsas_diam_vals_alldates = add_to_alldates_array(bin_diams[uhsas_setname], \
                                    uhsas_diam_vals_alldates)

        #make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
        #                                uhsas_aero_size_distb, \
        #                                date, n_samples, \
        #                                dN_CPC0, dN_CPC3)

    if n_samples == -1: #tbh this plot only makes sense for 1 curve per date
        make_and_save_aero_size_distb_plot_alldates(fan_aero_size_distb, \
                                uhsas_aero_size_distb_alldates, \
                                uhsas_diam_vals_alldates, n_samples, \
                                dN_CPC0_alldates, dN_CPC3_alldates)

def get_uhsas_setname(date):
    
    if date in ['20140916', '20140918', '20140919', '20140921']:
        return 'UHSAS2'
    else:
        return 'UHSAS'

def get_fan_aero_size_distb():

    #get aerosol psd extracted from fig S5C of fan et al 2018
    fan_aero_size_distb = []
    with open(DATA_DIR + 'fan_aero_size_distb.csv', 'r') as readFile:
        csvreader = csv.reader(readFile, \
                quoting=csv.QUOTE_NONNUMERIC, delimiter=',')
        for row in csvreader:
            fan_aero_size_distb.append(row)
    fan_aero_size_distb = np.array(fan_aero_size_distb)
    fan_aero_size_distb = \
        fan_aero_size_distb[fan_aero_size_distb[:,0].argsort()]

    return fan_aero_size_distb

def add_to_alldates_array(aero_size_distb, aero_size_distb_alldates):

    #if aero_size_distb_alldates is None:
    #    return aero_size_distb
    #else:
    #    return np.concatenate((aero_size_distb_alldates, aero_size_distb))
    if aero_size_distb_alldates is None:
        return [aero_size_distb]
    else:
        aero_size_distb_alldates.append(aero_size_distb)
        return aero_size_distb_alldates

def get_aero_size_distb(asd_data, setname, filter_inds, n_samples):
        
    diams = bin_diams[setname]
    diam_diffls = bin_diam_diffls[setname]
    # n_samples = # consecutive data pts to average in a row per asd curve
    # if n_samples is negative then a single curve (average over all pts)
    # is plotted
    if n_samples > 0:
        n_curves = ceil(np.sum(filter_inds)/n_samples)
    else:
        n_curves = 1
        n_samples = np.sum(filter_inds)

    aero_size_distb = np.zeros((n_curves, n_bins[setname]))

    for i in range(n_bins[setname]):
        var_key = 'nconc_' + str(i+bin_start_inds[setname])
        dN = np.array([np.nanmean( \
                asd_data[var_key][filter_inds][j:j+n_samples]) \
                for j in range(np.sum(filter_inds)) if j%n_samples == 0])
        dNdlogDp = 1./2.3025*dN*diams[i]/diam_diffls[i] # num/m^3
        aero_size_distb[:, i] = dNdlogDp
    
    return aero_size_distb
    
def make_and_save_aero_size_distb_plot(fan_aero_size_distb, \
                                        uhsas_aero_size_distb, \
                                        date, n_samples, dN_CPC0, dN_CPC3):

        uhsas_setname = get_uhsas_setname(date)

        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.plot(bin_diams[uhsas_setname]*1.e9, 1.e-6*uhsas_aero_size_distb.T, \
                linewidth=4)
        plt.gca().set_prop_cycle(None)
        ax.plot(cpc0_xrange*1.e9, \
                1.e-6*np.array([dN_CPC0/dlogDp_CPC0, dN_CPC0/dlogDp_CPC0]), \
                linewidth=4, \
                linestyle=':')
        plt.gca().set_prop_cycle(None)
        ax.plot(cpc3_xrange*1.e9, \
                1.e-6*np.array([dN_CPC3/dlogDp_CPC3, dN_CPC3/dlogDp_CPC3]), \
                linewidth=4, \
                linestyle='--')
        ax.plot(np.power(10, fan_aero_size_distb[:, 0]), \
                fan_aero_size_distb[:, 1], \
                label='Fan WRF simulation - polluted case', \
                color=colors['polluted'], \
                linewidth=6)
        ax.plot(np.power(10, fan_aero_size_distb[unpoll_cutoff:, 0]), \
                fan_aero_size_distb[unpoll_cutoff:, 1], \
                label='Fan WRF simulation - unpolluted case', \
                color=colors['unpolluted'], \
                linewidth=6, \
                linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel('diam (nm)')
        ax.set_ylabel('dN/dlogD (cm^-3)')
        ax.set_title(date + ' aerosol size distb' \
                        + ', change_cas_corr=' + str(change_cas_corr) \
                        + ', cutoff_bins=' + str(cutoff_bins) \
                        + ', n samples per curve=' + str(n_samples))
        #quick lazy dirty
        if n_samples == -1 and date == 'alldates':
            handles, labels = ax.get_legend_handles_labels()
            labels = ['20140909', '20140911', '20140912', '20140916', \
                        '20140918', '20140927', '20140928', '20140930', \
                        '20141001'] + labels
            ax.legend(labels)
        else:
            ax.legend()
        outfile = FIG_DIR + versionstr + 'aero_with_cpc_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

def make_and_save_aero_size_distb_plot_alldates(fan_aero_size_distb, \
                                uhsas_aero_size_distb_alldates, \
                                uhsas_diam_vals_alldates, n_samples, \
                                dN_CPC0_alldates, dN_CPC3_alldates):

        dates = ['20140909', '20140911', '20140912', '20140916', \
                '20140918', '20140927', '20140928', '20140930', '20141001']

        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        for i, uhsas_asd in enumerate(uhsas_aero_size_distb_alldates):
            ax.plot(uhsas_diam_vals_alldates[i]*1.e9, \
                    1.e-6*uhsas_asd.T, \
                    color=mplcycle[i], \
                    label=dates[i], \
                    linewidth=4)
        for i, dN_CPC0 in enumerate(dN_CPC0_alldates):
            ax.plot(cpc0_xrange*1.e9, \
                    1.e-6*np.array([dN_CPC0/dlogDp_CPC0, dN_CPC0/dlogDp_CPC0]), \
                    color=mplcycle[i], \
                    linewidth=4, \
                    label='_nolegend_', \
                    linestyle=':')
        #for i, dN_CPC3 in enumerate(dN_CPC3_alldates):
            #ax.plot(cpc3_xrange*1.e9, \
                    #1.e-6*np.array([dN_CPC3/dlogDp_CPC3, dN_CPC3/dlogDp_CPC3]), \
                    #color=mplcycle[i], \
                    #linewidth=4, \
                    #label='_nolegend_', \
                    #linestyle='--')
        ax.plot(np.power(10, fan_aero_size_distb[:, 0]), \
                fan_aero_size_distb[:, 1], \
                label='Fan WRF simulation - polluted case', \
                color=colors['polluted'], \
                linewidth=6)
        ax.plot(np.power(10, fan_aero_size_distb[unpoll_cutoff:, 0]), \
                fan_aero_size_distb[unpoll_cutoff:, 1], \
                label='Fan WRF simulation - unpolluted case', \
                color=colors['unpolluted'], \
                linewidth=6, \
                linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel('diam (nm)')
        ax.set_ylabel('dN/dlogD (cm^-3)')
        ax.set_title('alldates aerosol size distb' \
                        + ', change_cas_corr=' + str(change_cas_corr) \
                        + ', cutoff_bins=' + str(cutoff_bins) \
                        + ', n samples per curve=' + str(n_samples))
        ax.legend()
        outfile = FIG_DIR + versionstr + 'aero_with_cpc_' \
                + 'alldates_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)    

def get_residual_nconc(uhsas_data, cpc_data, filter_inds, n_samples, uhsas_setname):

    (uhsas_inds, cpc_inds) = match_two_arrays(uhsas_data['time'], \
                                                cpc_data['time'])

    filter_inds = filter_inds[uhsas_inds]

    uhsas_tot_nconc = sample_mean(get_uhsas_tot_nconc(uhsas_data, uhsas_inds, \
                                            filter_inds, uhsas_setname), \
                                            n_samples, filter_inds)
    print(uhsas_tot_nconc)
    cpc0_tot_nconc = sample_mean( \
            cpc_data['cpc0_nconc'][cpc_inds][filter_inds], \
            n_samples, filter_inds)
    print(cpc0_tot_nconc)
    cpc3_tot_nconc = sample_mean( \
            cpc_data['cpc3_nconc'][cpc_inds][filter_inds], \
            n_samples, filter_inds)
    print(cpc3_tot_nconc)

    return(cpc0_tot_nconc - uhsas_tot_nconc, cpc3_tot_nconc - uhsas_tot_nconc)

def get_uhsas_tot_nconc(uhsas_data, uhsas_inds, filter_inds, uhsas_setname):

    uhsas_tot_nconc = np.zeros(\
            np.shape(uhsas_data['time'][uhsas_inds][filter_inds]))

    start_ind = bin_start_inds[uhsas_setname]
    for i in range(start_ind, start_ind + n_bins[uhsas_setname]):
        bin_key = 'nconc_' + str(i) 
        uhsas_tot_nconc += uhsas_data[bin_key][uhsas_inds][filter_inds]

    return uhsas_tot_nconc

def sample_mean(array, n_samples, filter_inds):

    averaged_array = np.array([np.nanmean(array[j:j+n_samples]) \
                                for j in range(np.sum(filter_inds)) \
                                if j%n_samples == 0])

    return averaged_array
if __name__ == "__main__":
    main()
