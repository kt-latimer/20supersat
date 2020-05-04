"""
Count how many subsaturated updrafts are found horizontally adjacent to
(super)saturated ones. Realized later that this doesn't technically tell us
whether they are at the cloud edges but haven't refigured yet because it
already took a while to run as-is. Output format: number of subsat updrafts...
(with nearest horizontal neighbor (super)sat, with nnhn ss, with nnnhn ss, with
nnnnhn ss).
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-5
versionstr = 'v1_'

#plot stuff
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

def main():
    """
    For both polluted and unpolluted model runs, run point count and print to
    slurm output file results as described in module doc.
    """
    for model_label in model_dirs.keys():

        if model_label == 'Polluted':
            continue

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_ss_v4', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lh_K_s = ncsecvars['lh_K_s'][...]
        lwc = ncsecvars['lwc_cloud'][...]
        ss_qss = ncsecvars['ss_qss'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]
        #lh_K_s = ncsecvars['lh_K_s'][0, 42, :, :]
        #lwc = ncsecvars['lwc_cloud'][0, 42, :, :]
        #ss_qss = ncsecvars['ss_qss'][0, 42, :, :]
        #ss_wrf = ncsecvars['ss_wrf'][0, 42, :, :]
        #temp = ncsecvars['temp'][0, 42, :, :]
        #w = ncsecvars['w'][0, 42, :, :]

        ncsecfile.close()
        
        #make filter mask
        #mask = LWC_C > lwc_cutoff
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (nconc > 3.e6)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1), \
        #                            (np.abs(w) < 10)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC_C > lwc_cutoff), \
        #                            (np.abs(w) > 5)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        mask1 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (ss_wrf > 0), \
                                    (w > 2)))
        mask2 = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (ss_wrf < 0), \
                                    (w > 2)))
        
        nb_ct = [0, 0, 0, 0, 0]

        (inds0, inds1, inds2, inds3) = np.where(mask1)
        
        print(len(inds0))

        mask1_true_inds = [(inds0[j], inds1[j], inds2[j], inds3[j]) \
                            for j in range(len(inds0))]

        (inds0, inds1, inds2, inds3) = np.where(mask2)
        
        print(len(inds0))
        
        return

        mask2_true_inds = [(inds0[j], inds1[j], inds2[j], inds3[j]) \
                            for j in range(len(inds0))]

        for ind in mask2_true_inds:
            found_nb = False
            for level in range(5):
                for nb_ind in get_nb_inds(ind, level):
                    if nb_ind in mask1_true_inds:
                        nb_ct[level] += 1
                        found_nb = True
                        break
                if found_nb:
                    break
        
        print(model_label)
        print(nb_ct)
        print()

def get_nb_inds(ind, level):
    deltas = [[(0, 1), (0, -1), (1, 0), (-1, 0)], \
                [(1, 1), (1, -1), (-1, 1), (-1, -1)], \
                [(0, 2), (0, -2), (2, 0), (-2, 0)], \
                [(1, 2), (2, 1), (2, -1), (1, -2), \
                    (-1, -2), (-2, -1), (-2, 1), (-1, 2)], \
                [(2, 2), (2, -2), (-2, 2), (-2, -2)]]
    
    return [(ind[0], ind[1], ind[2]+d[0], ind[3]+d[1]) for d in deltas[level]]

if __name__ == "__main__":
    main()
