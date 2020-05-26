"""
plot constituent quantities of ss_qss vs ss_wrf to try to understand "two
prong" appearance of ss_wrf vs ss_qss plot
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
versionstr = 'v2_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'qty1': '#88720A', 'qty2': '#BA3F00', 'qty3': '#095793'}

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
    For both polluted and unpolluted model runs, plot constituent quantities of
    qss supersat vs wrf supersat.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir +
                            'wrfout_d01_secondary_vars_with_rain_and_vent', 'r')
        ncsecvars = ncsecfile.variables

        #get secondary variables
        lh_K_s = ncsecvars['lh_K_s'][...]
        #lh_J_m3_s = ncsecvars['lh_J_m3_s'][...]
        lwc = ncsecvars['lwc_cloud'][...]
        meanfr = ncsecvars['meanfr'][...]
        nconc = ncsecvars['nconc'][...]
        pres = ncsecvars['pres'][...]
        rho_air = ncsecvars['rho_air'][...]
        ss_ry = ncsecvars['ss_ry'][...]
        #ss_wrf = ncsecvars['ss_wrf'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        #ncprimfile.close()
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
        mask = np.logical_and.reduce(( \
                                    (lwc > lwc_cutoff), \
                                    (temp > 273), \
                                    (w > 2)))
        #mask = np.logical_and(mask, ss_wrf < 0)
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (np.abs(w) > 4)))
        #mask = np.logical_and.reduce(( \
        #                            (lwc > lwc_cutoff), \
        #                            (temp > 273), \
        #                            (w > 4)))
       
        lh_K_s = lh_K_s[mask]
        #lh_J_m3_s = lh_J_m3_s[mask]
        meanfr = meanfr[mask]
        nconc = nconc[mask]
        pres = pres[mask]
        rho_air = rho_air[mask]
        ss_ry = ss_ry[mask]
        temp = temp[mask]
        w = w[mask]
        
        del lwc, mask #for memory

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        F_d = rho_w*R_v*temp/(D*e_s)

        #quantities defined in ch 7 of Rogers and Yau
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        B = rho_w*(R_v*temp/e_s + R_a*L_v**2./(pres*temp*R_v*C_ap))

        del pres, temp, e_s #for memory

        qty1 = A*w/B
        qty2 = C_ap*rho_air/(L_v*rho_w)*lh_K_s
        qty3 = (F_k + F_d)/(4*np.pi*nconc*meanfr)

        del A, B, F_d, F_k, lh_K_s, meanfr, nconc, rho_air, w #for memory

        #do regression analysis
        #m, b, R, sig = linregress(qty1, qty2)
        #print(m, b, R**2)
        
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots(3, 2)
        ax[0][0].scatter(qty3, ss_ry, c=colors['qty3'])
        ax[0][0].set_xlim(np.array([np.min(qty3), np.max(qty3)]))
        ax[0][0].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[0][0].set_xlabel(r'$\frac{F}{4\pi n \langle f(r) \cdot r \rangle}$ (m$^2$/s)')
        ax[0][0].set_ylabel(r'$SS_{RY}$ (%)')
        ax[0][1].scatter(np.log10(qty3), ss_ry, c=colors['qty3'])
        ax[0][1].set_xlim(np.array([np.min(np.log10(qty3)), np.max(np.log10(qty3))]))
        ax[0][1].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[0][1].set_xlabel(r'$\log(\frac{F}{4\pi n \langle f(r) \cdot r \rangle})$')
        ax[0][1].set_ylabel(r'$SS_{RY}$ (%)')
        ax[1][0].scatter(qty1, ss_ry, c=colors['qty1'])
        ax[1][0].set_xlim(np.array([np.min(qty1), np.max(qty1)]))
        ax[1][0].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[1][0].set_xlabel(r'$\frac{A\cdot w}{B}$ (s/m$^2$)')
        ax[1][0].set_ylabel(r'$SS_{RY}$ (%)')
        ax[1][1].scatter(np.log10(qty1), ss_ry, c=colors['qty1'])
        ax[1][1].set_xlim(np.array([np.min(np.log10(qty1)), np.max(np.log10(qty1))]))
        ax[1][1].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[1][1].set_xlabel(r'$\log(\frac{A\cdot w}{B})$')
        ax[1][1].set_ylabel(r'$SS_{RY}$ (%)')
        ax[2][0].scatter(qty2, ss_ry, c=colors['qty2'])
        ax[2][0].set_xlim(np.array([np.min(qty2), np.max(qty2)]))
        ax[2][0].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[2][0].set_xlabel(r'$\frac{C_{ap}\rho_a LH}{L_v\rho_L}$ (s/m$^2$)')
        ax[2][0].set_ylabel(r'$SS_{RY}$ (%)')
        ax[2][1].scatter(np.log10(np.abs(qty2)), ss_ry, c=colors['qty2'])
        ax[2][1].set_xlim(np.array([np.min(np.log10(np.abs(qty2))), np.max(np.log10(qty2))]))
        ax[2][1].set_ylim(np.array([np.min(ss_ry), np.max(ss_ry)]))
        ax[2][1].set_xlabel(r'$\log(|\frac{C_{ap}\rho_a LH}{L_v\rho_L}|)$')
        ax[2][1].set_ylabel(r'$SS_{RY}$ (%)')
        fig.set_size_inches(21, 18)
        plt.tight_layout()

        outfile = FIG_DIR + versionstr + 'inclrain_and_vent_fancy_breakdown_qss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        del qty1, qty2, qty3 #for memory

if __name__ == "__main__":
    main()
