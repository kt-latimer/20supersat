"""
Create and save figure set comparelwcbycloud.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import calc_lwc

casbinfile = DATA_DIR + 'CAS_bins.npy'
CAS_bins = np.load(casbinfile, allow_pickle=True)

cdpbinfile = DATA_DIR + 'CDP_bins.npy'
CDP_bins = np.load(cdpbinfile, allow_pickle=True)

matplotlib.rcParams.update({'font.size': 22})

def main():
    """
    for a single date (specified manually for now), identify cloud instances \
    using LWC cutoff of 10^-5 for both CAS and CDP, compare cloud drop \
    size distributions for each cloud instance and save indiv. figs.
    """
    
    #load datasets and calculate lwc for CAS and CDP
    date = '20141001'

    adlrdatafile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
    adlrdata = np.load(adlrdatafile, allow_pickle=True).item()
    
    casdatafile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
    casdata = np.load(casdatafile, allow_pickle=True).item()
    caslwc = casdata['data']['lwc']['11']
    caslwctinds = casdata['data']['lwc_t_inds']
    cast = casdata['data']['time'][caslwctinds]

    cdpdatafile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
    cdpdata = np.load(cdpdatafile, allow_pickle=True).item()
    cdplwc = cdpdata['data']['lwc']['11']
    cdplwctinds = cdpdata['data']['lwc_t_inds']
    cdpt = cdpdata['data']['time'][cdplwctinds]
    
    #get cas clouds
    cas_cloud_clusters = []
    current_cloud = []
    in_cloud = False
    for i, val in enumerate(caslwc):
        if val > 1.e-5:
            if in_cloud:
                current_cloud.append(i)
            else:
                in_cloud = True
                current_cloud.append(i)
        else:
            if in_cloud:
                cas_cloud_clusters.append(current_cloud)
                current_cloud = []
                in_cloud = False
    print('cas')
#    print(len(cas_cloud_clusters))
#    for i, cluster in enumerate(cas_cloud_clusters[:-1]):
#        print(len(cluster), cas_cloud_clusters[i+1][0] - cluster[-1])

    big_cas_clusters = []
    big_cluster = cas_cloud_clusters[0]
    for i, cluster in enumerate(cas_cloud_clusters[:-1]):
        if cast[cas_cloud_clusters[i+1][0]] - cast[cluster[-1]] < 30:
            big_cluster += cas_cloud_clusters[i+1]
        else:
            big_cas_clusters.append(big_cluster)
            big_cluster = cas_cloud_clusters[i+1]
    print(len(big_cas_clusters))
    for thing in big_cas_clusters[:-1]:
        print(len(thing), cast[thing[0]], cast[thing[-1]], thing[0], thing[-1])

    #get cdp clouds
    cdp_cloud_clusters = []
    current_cloud = []
    in_cloud = False
    for i, val in enumerate(cdplwc):
        if val > 1.e-5:
            if in_cloud:
                current_cloud.append(i)
            else:
                in_cloud = True
                current_cloud.append(i)
        else:
            if in_cloud:
                cdp_cloud_clusters.append(current_cloud)
                current_cloud = []
                in_cloud = False
    print('cdp')
#    print(len(cdp_cloud_clusters))
#    for i, cluster in enumerate(cdp_cloud_clusters[:-1]):
#        print(len(cluster), cdp_cloud_clusters[i+1][0] - cluster[-1])
    
    big_cdp_clusters = []
    big_cluster = cdp_cloud_clusters[0]
    for i, cluster in enumerate(cdp_cloud_clusters[:-1]):
        if cdpt[cdp_cloud_clusters[i+1][0]] - cdpt[cluster[-1]] < 30:
            big_cluster += cdp_cloud_clusters[i+1]
        else:
            big_cdp_clusters.append(big_cluster)
            big_cluster = cdp_cloud_clusters[i+1]
    print(len(big_cdp_clusters))
    for thing in big_cdp_clusters[:-1]:
        print(len(thing), cdpt[thing[0]], cdpt[thing[-1]], thing[0], thing[-1])

    #possibly useful generic lines:
    #
    #fig, ax = plt.subplots()
    #...or:
    #fig=plt.gcf() 
    #
    #fig.set_size_inches(21, 12)
    outfile = FIG_DIR + 'comparelwcbycloud_set_figsrc.png'
    plt.savefig(outfile)

def cloud_pdf_compare(cas_cloud, cdp_cloud, cloudlabel):
    print('foo bar')
if __name__ == "__main__":
    main()
