"""
Create and save figure meanr_scatter_set.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.cloudevents_set_figsrc import get_datablock, get_meanr_vs_t
from halo.utils import get_ind_bounds, match_multiple_arrays

matplotlib.rcParams.update({'font.size': 14})

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

def main():
	"""
	the main routine.
	"""
	dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
	     '20140918', '20140921', '20140927', '20140928', \
	     '20140930','20141001']
	offsets = [0, 2, 2, 0, 0, 0, 0, 1, 0, 1, 3]

	for m, date in enumerate(dates[0:1]):
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
			[np.fix(adlrdata['data']['time']), \
			np.fix(casdata['data']['time']), \
			np.fix(cdpdata['data']['time'])])
		datablock = get_datablock(adlrinds, casinds, cdpinds, \
			adlrdata, casdata, cdpdata)

		#remove rows with error values in any of the three
		goodrows = []
		for i, row in enumerate(datablock):
			if sum(np.isnan(np.concatenate((row[0:2], row[3:])))) == 0:
				goodrows.append(i)
		n = len(goodrows)
		nerr = np.shape(datablock)[0] - n
		datablock = datablock[goodrows, :]
		(cas_meanr, cdp_meanr) = get_meanr_vs_t(datablock)
		cas_meanr = cas_meanr*1.e6
		cdp_meanr = cdp_meanr*1.e6
		if m == 0:
			all_cas = cas_meanr
			all_cdp = cdp_meanr
		else:
			all_cas = np.concatenate((all_cas, cas_meanr))
			all_cdp = np.concatenate((all_cdp, cdp_meanr))
		fig, ax = plt.subplots()
		ax.scatter(cas_meanr, cdp_meanr)
	#	coef = np.polyfit(cas_meanr, cdp_meanr, 1)
	#	print(coef)
	#	poly1d_fn = np.poly1d(coef) 
	#	ax.plot(cas_meanr, poly1d_fn(cas_meanr), '--k')
	#	ax.text(0.5, 0.5, str(coef[0]), horizontalalignment='center', \
	#		verticalalignment='center', transform=ax.transAxes)
		ax.set_xlabel('CAS')
		ax.set_ylabel('CDP')
		ax.set_title('Mean particle radius (um)')
		ax.set_aspect('equal', 'datalim')
		fig.set_size_inches(21, 12)
		outfile = FIG_DIR + date + '_meanr_scatter_figure.png'
		plt.savefig(outfile)

	fig, ax = plt.subplots()
	ax.scatter(all_cas, all_cdp)
#	coef = np.polyfit(all_cas, all_cdp, 1)
#	print(coef)
#	poly1d_fn = np.poly1d(coef) 
#	ax.plot(all_cas, poly1d_fn(all_cas), '--k')
#	ax.text(0.5, 0.5, str(coef[0]), horizontalalignment='center', \
#		verticalalignment='center', transform=ax.transAxes)
	ax.set_xlabel('CAS')
	ax.set_ylabel('CDP')
	ax.set_title('Mean particle radius (um)')
	ax.set_aspect('equal', 'datalim')
	fig.set_size_inches(21, 12)
	outfile = FIG_DIR + 'meanr_scatter_set_figure.png'
	plt.savefig(outfile)

if __name__ == "__main__":
    main()
