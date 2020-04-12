"""
Create and save figure qss_vs_wrf. This is a scatter plot comparing WRF's
supersaturation output against a simplified version of the quasi-steady-state
supersaturation equation in Korolev 2003.
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset, MFDataset
import numpy as np

from halo.utils import linregress
from mywrf import BASE_DIR, DATA_DIR, FIG_DIR 

model_dirs = {'Polluted':'C_BG/', 'Unpolluted':'C_PI/'}
lwc_cutoff = 5.e-5
versionstr = 'v50_'
cutoff_x_axis = False 

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
    For both polluted and unpolluted model runs, plot qss SS approx vs WRF SS.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        ncsecfile = Dataset(DATA_DIR + model_dir + 'wrfout_d01_secondary_vars', 'r')
        ncsecvars = ncsecfile.variables
        
        #get relevant primary variables from wrf output
        LH = ncprimvars['TEMPDIFFL'][...]
        LWC = ncprimvars['QCLOUD'][...]
        PH = ncprimvars['PH'][...] #geopotential perturbation
        PHB = ncprimvars['PHB'][...] #geopotential base value
        SS = ncprimvars['SSW'][...]

        #get secondary variables
        meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        rho_air = ncsecvars['rho_air'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncprimfile.close()
        ncsecfile.close()

        #formula for saturation vapor pressure from Rogers and Yau - converted
        #to mks units (p 16)
        e_s = 611.2*np.exp(17.67*(temp - 273)/(temp - 273 + 243.5))
        
        #quantities defined in ch 7 of Rogers and Yau
        F_k = (L_v/(R_v*temp) - 1)*(L_v*rho_w/(K*temp))
        F_d = rho_w*R_v*temp/(D*e_s)
        #F = F_d + F_k

        #get altitude (geopotential) and realign to mass grid
        z = (PH + PHB)/g #altitude rel to sea level
        del PH, PHB
        z = (z[:, 0:-1, :, :] + z[:, 1:, :, :])/2

        ##first abstract quantity
        #qty_1 = rho_air*(F_k + F_d)*LH/(L_v*rho_w)

        #ss formula from RY p 102
        ss = C_ap*rho_air*(F_k + F_d)*LH/(4*np.pi*L_v*rho_w*meanr*nconc)

        #make filter mask
        #mask = LWC > lwc_cutoff
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (z < 6000), \
        #                            (meanr > 0), \
        #                            (meanr < 60.e-6)))
        mask = np.logical_and.reduce(( \
                                    (LWC > lwc_cutoff), \
                                    (meanr > 0), \
                                    (meanr < 30.e-6)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1), \
        #                            (np.abs(w) < 10)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1)))
        #
        #print(np.shape(mask))
        #print('num above lwc cutoff and nonzero w: ', np.sum(mask))
        #
        #do regression analysis
        m, b, R, sig = linregress(ss[mask]*100, SS[mask]*100)
        print(m, b, R**2)
        #
        ##count number of points outside range [-100, 100] for qss set
        #n_hi = np.sum(np.logical_and.reduce(( \
        #                            (ss > 1), \
        #                             mask)))
        #n_lo = np.sum(np.logical_and.reduce(( \
        #                            (ss < -1), \
        #                             mask)))
        #print(model_label)
        #print('n_hi: ', n_hi)
        #print('n_lo: ', n_lo)
        #
        #n_q1 = np.sum(np.logical_and.reduce(( \
        #                            (ss > 0), \
        #                            (SS > 0), \
        #                             mask)))
        #n_q2 = np.sum(np.logical_and.reduce(( \
        #                            (ss < 0), \
        #                            (SS > 0), \
        #                             mask)))
        #n_q3 = np.sum(np.logical_and.reduce(( \
        #                            (ss < 0), \
        #                            (SS < 0), \
        #                             mask)))
        #n_q4 = np.sum(np.logical_and.reduce(( \
        #                            (ss > 0), \
        #                            (SS < 0), \
        #                             mask)))
        #
        #print('Number of points in Q1:', n_q1)
        #print('Number of points in Q2:', n_q2)
        #print('Number of points in Q3:', n_q3)
        #print('Number of points in Q4:', n_q4)
        #print()
        #
        ###get limits of the data for plotting purposes
        ##xlim_max = np.max(np.array( \
        ##                [np.max(SS_qss[mask]), \
        ##                 np.max(SS_wrf[mask])]))
        ##xlim_min = np.min(np.array( \
        #                [np.min(SS_qss[mask]), \
        #                 np.min(SS_wrf[mask])]))
        #ax_lims = np.array([100*xlim_min, 100*xlim_max])
        #print(ax_lims)
        
        x_ax_lims = np.array([-1000, 1000])
        #x_ax_lims = np.array([-1, 1])
        y_ax_lims = np.array([-100, 100])
        
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        im = ax.scatter(ss[mask]*100, SS[mask]*100, c=z[mask], cmap='coolwarm')#c=colors['ss'])
        #im = ax.scatter(LH[mask]*C_ap, SS[mask]*100, c=temp[mask], cmap='coolwarm')#c=colors['ss'])
        ax.plot(x_ax_lims, np.add(b, m*x_ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        #ax.set_aspect('equal', 'box')
        if cutoff_x_axis:
            ax.set_xlim(x_ax_lims)
        else:
            if model_label == 'Unpolluted':
                #there's a couple weird outliers above 1.e6 
                ax.set_xlim(np.array([-1.e5, 1.e6]))
        ax.set_ylim(y_ax_lims)
        ax.set_xlabel('RY supersat (%)')
        ax.set_ylabel('WRF supersat (%)')
        fig.legend()
        fig.set_size_inches(21, 12)
        #fig.colorbar(im, ax=ax, label='Mean radius (m)')
        #fig.colorbar(im, ax=ax, label='Num. conc. (m^-3)')
        #fig.colorbar(im, ax=ax, label='F_k + F_d (s m^-2)')
        #fig.colorbar(im, ax=ax, label='LH (K/s)')
        #fig.colorbar(im, ax=ax, label='Air dens. (kg m^-3)')
        #fig.colorbar(im, ax=ax, label='w (m/s)')
        fig.colorbar(im, ax=ax, label='Altitude (m)')
        
        outfile = FIG_DIR + versionstr + 'backtrack_qss_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        del LH, LWC, SS, meanr, nconc, rho_air, temp, w, e_s, F_k, F_d
        del ss, mask
        del z

if __name__ == "__main__":
    main()
