"""
Same as inclrain_backtrack_qss_figsrc but with ventilation correction 
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
versionstr = 'v4_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
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
    For both polluted and unpolluted model runs, plot some quantity described
    in the notes.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lh_K_s = ncsecvars['lh_K_s'][...]
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        rho_air = ncsecvars['rho_air'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]
        
        ncsecfile.close()

        #sat vapor pressure
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        F_d = rho_w*R_v*temp/(D*e_s)

        #ss formula from RY p 102
        ss_ry = C_ap*rho_air \
            *(F_k + F_d)*lh_K_s/(4*np.pi*L_v*rho_w*meanfr*nconc)
        
        del e_s, F_d, F_k #for memory
        
        #make filter mask
        #mask = LWC_C > lwc_cutoff
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (nconc > 3.e6)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1), \
        #                            (np.abs(w) < 10)))
        mask = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (lh_K_s > 0)))
        
        print(model_label)
        ratio = meanfr[mask]/meanr[mask]
        print('ratio mean: ', np.nanmean(ratio))
        print('ratio median: ', np.nanmedian(ratio))
        print('ratio std: ', np.nanstd(ratio))
        print('ratio max: ', np.nanmax(ratio))
        print('ratio min: ', np.nanmin(ratio))
       
        continue 

        print(np.shape(mask))
        print('num above lwc cutoff: ', np.sum(mask))
        
        #do regression analysis
        m, b, R, sig = linregress(ss_ry[mask]*100, ss_wrf[mask]*100)
        print(m, b, R**2)
        
        #count number of points outside range [-100, 100] for qss_ry set
        n_hi = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 1), \
                                     mask)))
        n_lo = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < -1), \
                                     mask)))
        print(model_label)
        print('n_hi: ', n_hi)
        print('n_lo: ', n_lo)
        print('max ssry:', np.nanmax(ss_ry))
        print('min ssry:', np.nanmin(ss_ry))
        
        n_q1 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 0), \
                                    (ss_wrf > 0), \
                                     mask)))
        n_q2 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < 0), \
                                    (ss_wrf > 0), \
                                     mask)))
        n_q3 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry < 0), \
                                    (ss_wrf < 0), \
                                     mask)))
        n_q4 = np.sum(np.logical_and.reduce(( \
                                    (ss_ry > 0), \
                                    (ss_wrf < 0), \
                                     mask)))
        
        print('Number of points in Q1:', n_q1)
        print('Number of points in Q2:', n_q2)
        print('Number of points in Q3:', n_q3)
        print('Number of points in Q4:', n_q4)
        print()
        
        ##get limits of the data for plotting purposes
        #xlim_max = np.max(np.array( \
        #                [np.max(ss_wrf_qss[mask]), \
        #                 np.max(ss_wrf_wrf[mask])]))
        #xlim_min = np.min(np.array( \
        #                [np.min(ss_wrf_qss[mask]), \
        #                 np.min(ss_wrf_wrf[mask])]))
        #ax_lims = np.array([100*xlim_min, 100*xlim_max])
        #print(ax_lims)
        
        ax_lims = np.array([-100, 100])
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(ss_ry[mask]*100, ss_wrf[mask]*100, c=colors['ss'])
        ax.plot(ax_lims, np.add(b, m*ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xlabel(r'$SS_{RY}$ (%)')
        ax.set_ylabel(r'$SS_{WRF}$ (%)')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_backtrack_qss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
