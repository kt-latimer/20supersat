"""
Same as inclrain_qss_vs_fan but with ventilation correction 
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

PROJ_DATA_DIR = '/global/home/users/kalatimer/proj/20supersat/data/mywrf/'

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-5 
versionstr = 'v15_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'bulk': '#095793', 'edge': '#88720A'}

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
    For both polluted and unpolluted model runs, plot qss vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        #ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        #ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
                            #'wrfout_d01_secondary_vars_no_rain', 'r')
        ncsecvars = ncsecfile.variables

        #get primary variables
        #q_ice = ncprimvars['QICE'][...]
        #PH = ncprimvars['PH'][...] #geopotential perturbation
        #PHB = ncprimvars['PHB'][...] #geopotential base value
        
        #z = (PH + PHB)/g #altitude rel to sea level
        #z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2

        #del PH, PHB #for memory
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        #meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        #pres = ncsecvars['pres'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]
        #lwc = ncsecvars['lwc_cloud'][...][0:-1, 0:-1, 0:-1, 0:-1]
        #meanfr = ncsecvars['meanfr'][...][0:-1, 0:-1, 0:-1, 0:-1]
        #nconc = ncsecvars['nconc'][...][0:-1, 0:-1, 0:-1, 0:-1]
        ##pres = ncsecvars['pres'][...]
        #ss_wrf = ncsecvars['ss_wrf'][...][0:-1, 0:-1, 0:-1, 0:-1]
        #temp = ncsecvars['temp'][...]
        #w = ncsecvars['w'][...][0:-1, 0:-1, 0:-1, 0:-1]
        ##
        #delta_z = z[0:-1, 1:, 0:-1, 0:-1] - z[0:-1, 0:-1, 0:-1, 0:-1]
        #temp_derv = (temp[0:-1, 1:, 0:-1, 0:-1] \
        #            - temp[0:-1, 0:-1, 0:-1, 0:-1])/delta_z
        #filename = PROJ_DATA_DIR + 'temp_derv_' + model_label + '.npy'
        #np.save(filename, temp_derv.tolist())
        #continue
        #for j in range(temp_derv.shape[0]):
        #    for k in range(temp_derv.shape[1]):
        #        temp_derv[j, k, :, :] = np.mean(temp_derv[j, k, :, :])    
        #temp = temp[0:-1, 0:-1, 0:-1, 0:-1]
        
        #del delta_z, z #for memory

        #ncprimfile.close()
        ncsecfile.close()


        #lwc = lwc + q_ice

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        #e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau (B is slightly different
        #from Q_2 because)
        #B = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))
        #F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        #F_d = rho_w*R_v*temp/(D*e_s)
        
        #factor in denominator of Rogers and Yau qss ss formula (p 110)
        #denom = B/(F_k + F_d)

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        #A = (-1.*temp_derv*L_v*R_a/R_v*1./temp - g)*1./R_a*1./temp
        #A = -1.*temp_derv*L_v/(R_v*temp**2.)
        ss_qss = w*A/(4*np.pi*D*nconc*meanfr)
        #ss_qss = w*A/(4*np.pi*D*nconc*meanr)
        #ss_qss = w*A/(4*np.pi*denom*nconc*meanfr)
       
        #del temp_derv #for memory

        #del A, meanfr, nconc #for memory
        #del A, meanr, nconc #for memory
        #del e_s, B, F_d, F_k, denom, A, q_ice, pres, nconc #for memory
        #make filter mask
        #mask = LWC_C > lwc_cutoff
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (nconc > 3.e6)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1), \
        #                            (np.abs(w) < 10)))
        bulk_mask = np.array(np.zeros(lwc.shape), dtype=bool)
        edge_mask = np.array(np.zeros(lwc.shape), dtype=bool)
        for j in range(lwc.shape[1]): #all z coords
            mask_z = np.logical_and.reduce(( \
                                    (lwc[:, j, :, :] > lwc_cutoff), \
                                    (temp[:, j, :, :] > 273), \
                                    (w[:, j, :, :] > 2)))
            if np.sum(mask_z) != 0:
                lwc_z = lwc[:, j, :, :][mask_z]
                cutoff_z = np.percentile(lwc_z, 5)
                bulk_mask[:, j, :, :] = np.logical_and(mask_z, \
                                    lwc[:, j, :, :] > cutoff_z)
                edge_mask[:, j, :, :] = np.logical_and(mask_z, \
                                    lwc[:, j, :, :] < cutoff_z)
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 2)))
                                    #(w > 1)))
        #new_lwc_cutoff = np.percentile(lwc, 10)
        #mask = np.logical_and(mask, lwc > new_lwc_cutoff)
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
        #ss_qss = ss_qss[mask]
        #ss_wrf = ss_wrf[mask]
        #lwc = lwc[mask]

        #edge_mask = lwc < np.percentile(lwc, 5)
        #ss_bulk = ss_wrf[np.logical_not(edge_mask)]
        #ss_edge = ss_wrf[edge_mask]
        ss_bulk = ss_wrf[bulk_mask]
        ss_edge = ss_wrf[edge_mask]
        
        #print(np.shape(mask))
        #print('num above lwc cutoff: ', np.sum(mask))
        #
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        #ax.hist(ss_bulk*100, bins = 40, color=colors['bulk'], alpha=0.7, \
        #        label='Cloud bulk', density=True)
        #ax.hist(ss_edge*100, bins = 40, color=colors['edge'], alpha=0.7, \
        #        label='Cloud edge', density=True)
        #ax.hist(ss_bulk*100, bins = 40, color=colors['bulk'], alpha=0.7, \
        #        label='Cloud bulk')
        #ax.hist(ss_edge*100, bins = 40, color=colors['edge'], alpha=0.7, \
        #        label='Cloud edge')
        #ax.hist(ss_bulk*100, bins = 40, color=colors['bulk'], alpha=0.7, \
        #        label='Cloud bulk', density=True, log=True)
        #ax.hist(ss_edge*100, bins = 40, color=colors['edge'], alpha=0.7, \
        #        label='Cloud edge', density=True, log=True)
        ax.hist(ss_bulk*100, bins = 40, color=colors['bulk'], alpha=0.7, \
                label='Cloud bulk', log=True)
        ax.hist(ss_edge*100, bins = 40, color=colors['edge'], alpha=0.7, \
                label='Cloud edge', log=True)
        #ax.set_xlabel(r'$SS_{QSS}$ (%)')
        ax.set_xlabel(r'$SS_{WRF}$ (%)')
        #ax.set_ylabel('Frequency')
        ax.set_ylabel('Count')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_bulkedge_hist_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
