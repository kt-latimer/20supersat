"""
Create and save figure nconc_scatter_set.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.cloudevents_set_figsrc import get_datablock, get_nconc_vs_t
from halo.utils import get_ind_bounds, linregress, match_multiple_arrays

#bin size data and settings depending on cutoff_bins param
#(indices are for columns of datablock variable)
casbinfile = DATA_DIR + 'CAS_bins.npy'
CAS_bins = np.load(casbinfile, allow_pickle=True).item()
cas_centr = (CAS_bins['upper'] + CAS_bins['lower'])/4. #diam to radius
cas_dr = CAS_bins['upper'] - CAS_bins['lower']
cas_nbins = len(cas_centr)

cdpbinfile = DATA_DIR + 'CDP_bins.npy'
CDP_bins = np.load(cdpbinfile, allow_pickle=True).item()
cdp_centr = (CDP_bins['upper'] + CDP_bins['lower'])/4. #diam to radius
cdp_dr = CDP_bins['upper'] - CDP_bins['lower']
cdp_nbins = len(cdp_centr)

low_bin_cas = 3
high_bin_cas = low_bin_cas + cas_nbins
low_bin_cdp = high_bin_cas 
high_bin_cdp = low_bin_cdp + cdp_nbins

if cutoff_bins:
	low_bin_cas = low_bin_cas + 3
	low_bin_cdp = low_bin_cdp + 2

#for plotting
colors = {'ADLR': '#777777', 'CAS': '#95B9E9', 'CDP': '#FC6A0C', 'c1': '#BA3F00', 'c2': '#095793'}

def main():
	"""
	the main routine.
	"""
	dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
	     '20140919', '20140918', '20140921', '20140927', '20140928', \
	     '20140930', '20141001']
	offsets = [0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 1, 3]

	for m, date in enumerate(dates):
		#load data
		adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
		adlrdata = np.load(adlrfile, allow_pickle=True).item()
		casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
		casdata = np.load(casfile, allow_pickle=True).item()
		casdata['data']['time'] = np.array([t - offsets[m] for t in casdata['data']['time']])
		cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
		cdpdata = np.load(cdpfile, allow_pickle=True).item()

		#loop through all cloud events and make a figure for each
		[adlrinds, casinds, cdpinds] = match_multiple_arrays(
			[np.around(adlrdata['data']['time']), \
			np.around(casdata['data']['time']), \
			np.around(cdpdata['data']['time'])])
		datablock_uncorr = get_datablock(adlrinds, casinds, cdpinds, \
			adlrdata, casdata, cdpdata, False)
		datablock_corr = get_datablock(adlrinds, casinds, cdpinds, \
			adlrdata, casdata, cdpdata, True)

		#remove rows with error values in any of the three
		goodrows = []
		for i, row in enumerate(datablock_corr):
			if sum(np.isnan(np.concatenate((row[0:2], row[3:])))) == 0:
				goodrows.append(i)
		datablock_corr = datablock_corr[goodrows, :]
		
		goodrows = []
		for i, row in enumerate(datablock_uncorr):
			if sum(np.isnan(np.concatenate((row[0:2], row[3:])))) == 0:
				goodrows.append(i)
		datablock_uncorr = datablock_uncorr[goodrows, :]
		
		:q!(cas_nconc, cdp_nconc) = get_nconc_vs_t(datablock)
		cas_nconc = cas_nconc*1.e-6
		cdp_nconc = cdp_nconc*1.e-6
		if m == 0:
			all_cas = cas_nconc
			all_cdp = cdp_nconc
		else:
			all_cas = np.concatenate((all_cas, cas_nconc))
			all_cdp = np.concatenate((all_cdp, cdp_nconc))
		fig, ax = plt.subplots()
		ax.scatter(cas_nconc, cdp_nconc)
		m, b, R, sig = linregress(cas_nconc_cdp_nconc)
		ax.text(0.5, 0.5, str(coef[0]), horizontalalignment='center', \
			verticalalignment='center', transform=ax.transAxes)
		ax.set_xlabel('CAS')
		ax.set_ylabel('CDP')
		ax.set_title('Total number concentration (cm^-3)')
		ax.set_aspect('equal', 'datalim')
		fig.set_size_inches(21, 12)
		outfile = FIG_DIR + date + '_nconc_scatter_figure.png'
		plt.savefig(outfile)

	fig, ax = plt.subplots()
	ax.scatter(all_cas, all_cdp)
	coef = np.polyfit(all_cas, all_cdp, 1)
	print(coef)
	poly1d_fn = np.poly1d(coef) 
	ax.plot(all_cas, poly1d_fn(all_cas), '--k')
	ax.text(0.5, 0.5, str(coef[0]), horizontalalignment='center', \
		verticalalignment='center', transform=ax.transAxes)
	ax.set_xlabel('CAS')
	ax.set_ylabel('CDP')
	ax.set_title('Total number concentration (cm^-3)')
	ax.set_aspect('equal', 'datalim')
	fig.set_size_inches(21, 12)
	outfile = FIG_DIR + 'nconc_scatter_set_figure.png'
	plt.savefig(outfile)

if __name__ == "__main__":
    main()

