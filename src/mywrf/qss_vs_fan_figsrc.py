"""
Create and save figure qss_vs_wrf. This is a scatter plot comparing WRF's
supersaturation output against a simplified version of the quasi-steady-state
supersaturation equation in Korolev 2003.
"""
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import MFDataset
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
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_w = 1000. #density of water (kg/m^3)

#wrf reference values
P_0 = 1.e5 #ref pressure (Pa)
th_0 = 300. #ref pot temp in (K)

def main():
    """
    For both polluted and unpolluted model runs, plot qss SS approx vs WRF SS.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load datafiles
        ncfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncvars = ncfile.variables
        
        ts('Start get prim vars')
        #get relevant variables from wrf output
        P_base = ncvars['PB'][...]
        P_pertb = ncvars['P'][...]
        LWC_wrf = ncvars['QCLOUD'][...]
        nconc_wrf = ncvars['QNCLOUD'][...]
        SS_wrf = ncvars['SSW'][...]
        th_pertb = ncvars['T'][...]
        W_wrf = ncvars['W'][...]
        #vertical wind velocity is on a staggered grid; take NN average to
        #reshape to mass grid
        W_wrf = (W_wrf[:,0:-1,:,:] + W_wrf[:,1:,:,:])/2
        ts('End get prim vars')

        ts('Start calc sec vars')
        #convert wrf outputs to real quantities
        th = th_0 + th_pertb
        P = P_base + P_pertb
        T = th*np.power((P/P_0), R_a/C_ap) #temperature (K)
        
        #calculate air density 
        rho_air = P/(R_a*T) #(kg/m^3)
        ts('End calc sec vars')

        ts('Start calc meanr')
        #compute mean radius of cloud droplets
        diams = [(2.**(1+i/3.))*10**(-6) for i in range(15)] #bin diams in m
        rads = [d/2. for d in diams] 
        rN_sum = np.empty(np.shape(P_base))
        N_sum = np.empty(np.shape(P_base))
        for i in range(15):
            r = rads[i]
            ff_i_wrf = ncvars['ff1i'+f'{i+1:02}'][...]
            N_i = ff_i_wrf*2.**i/(4./3.*np.pi*r**3.*rho_w/rho_air)
            N_sum += N_i
            rN_sum += N_i*r
        meanr = rN_sum/N_sum
        nconc = N_sum
        #nconc_kt = N_sum
        ts('End calc meanr')

        ts('Start calc qss ss')
        #compute quasi steady state ss (wrf gives cloud drop num conc in #/kg,
        #so need to convert to #/m^3 by factor of rho_air)
        A = g*(L_v*R_a/(C_ap*R_v)*1/T - 1)*1./R_a*1./T
        SS_qss = W_wrf*A/(4*np.pi*D*nconc*meanr)
        ts('End calc qss ss')
       
        ts('Start make mask')
        #make filter mask for LWC
        mask = LWC_wrf > lwc_cutoff
        ts('End make mask')

        ts('Start lin regress')
        m, b, R, sig = linregress(SS_qss[mask]*100, SS_wrf[mask]*100)
        print(m, b, R**2)
        ts('End lin regress')

        ts('Start min max')
        #get limits of the data for plotting purposes
        xlim_max = np.max(np.array( \
                        [np.max(SS_qss[mask]), \
                         np.max(SS_wrf[mask])]))
        xlim_min = np.min(np.array( \
                        [np.min(SS_qss[mask]), \
                         np.min(SS_wrf[mask])]))
        ax_lims = np.array([100*xlim_min, 100*xlim_max])
        print(ax_lims)
        ts('End min max')

        ts('Start plot')
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(SS_qss[mask]*100, SS_wrf[mask]*100, c=colors['ss'])
        ax.plot(ax_lims, np.add(b, m*ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_xlabel('Quasi steady state SS (%)')
        ax.set_ylabel('WRF SS (%)')
        fig.legend()
        fig.set_size_inches(21, 12)
        ts('End plot')

        ts('Start save')
        outfile = FIG_DIR + versionstr + 'qss_vs_fan_' \
                    + model_label + '_figure.png'
        plt.savefig(outfile)
        ts('End save')

def ts(message):
    """
    print time stamp along with message
    """
    print(message, datetime.now())
    return

if __name__ == "__main__":
    main()
