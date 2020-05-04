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

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 1.e-5
versionstr = 'v3_'

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
    For both polluted and unpolluted model runs, plot qss vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        if model_label == 'Polluted':
            continue

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables
        
        #get secondary variables
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        nconc = ncsecvars['nconc'][...]
        pres = ncsecvars['pres'][...]
        ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        Q_2 = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))
        F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        F_d = rho_w*R_v*temp/(D*e_s)
        
        #factor in denominator of Rogers and Yau qss ss formula (p 110)
        denom = Q_2/(F_k + F_d)

        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        #ss_qss = w*A/(4*np.pi*D*nconc*meanfr)
        ss_qss = w*A/(4*np.pi*denom*nconc*meanfr)
        
        del e_s, Q_2, F_d, F_k, denom, A
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
                                    (np.abs(w) > 2)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
        
        print(np.shape(mask))
        print('num above lwc cutoff: ', np.sum(mask))
        
        #do regression analysis
        m, b, R, sig = linregress(ss_qss[mask]*100, ss_wrf[mask]*100)
        print(m, b, R**2)
        
        #count number of points outside range [-100, 100] for qss_qss set
        n_hi = np.sum(np.logical_and.reduce(( \
                                    (ss_qss > 1), \
                                     mask)))
        n_lo = np.sum(np.logical_and.reduce(( \
                                    (ss_qss < -1), \
                                     mask)))
        print(model_label)
        print('n_hi: ', n_hi)
        print('n_lo: ', n_lo)
        print('ssqss max:', np.nanmax(ss_qss))
        print('ssqss min:', np.nanmin(ss_qss))
        
        n_q1 = np.sum(np.logical_and.reduce(( \
                                    (ss_qss > 0), \
                                    (ss_wrf > 0), \
                                     mask)))
        n_q2 = np.sum(np.logical_and.reduce(( \
                                    (ss_qss < 0), \
                                    (ss_wrf > 0), \
                                     mask)))
        n_q3 = np.sum(np.logical_and.reduce(( \
                                    (ss_qss < 0), \
                                    (ss_wrf < 0), \
                                     mask)))
        n_q4 = np.sum(np.logical_and.reduce(( \
                                    (ss_qss > 0), \
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
        ax.scatter(ss_qss[mask]*100, ss_wrf[mask]*100, c=colors['ss'])
        ax.plot(ax.get_xlim(), np.add(b, m*np.array(ax.get_xlim())), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xlabel(r'$SS_{QSS}$ (%)')
        ax.set_ylabel(r'$SS_{WRF}$ (%)')
        fig.legend(loc=2)
        fig.set_size_inches(21, 12)

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_qss_vs_fan_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

if __name__ == "__main__":
    main()
