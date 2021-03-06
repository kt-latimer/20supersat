"""
Create and save figure subregion_qss_vs_wrf. This is a scatter plot comparing 
WRF's supersaturation output against a simplified version of the 
quasi-steady-state supersaturation equation in Korolev 2003, for a selected 
subset of the total horizontal region.
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
versionstr = 'v5_'

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

def main():
    """
    For both polluted and unpolluted model runs, plot qss SS approx vs WRF SS
    for selected horz subregion.
    """
    for model_label in model_dirs.keys():

        model_dir = model_dirs[model_label]        

        #load secondary datafiles
        ncsecfile = Dataset(DATA_DIR + model_dir + 'wrfout_d01_secondary_vars', 'r')
        ncsecvars = ncsecfile.variables

        #get secondary variables
        meanr = ncsecvars['meanr'][...]
        nconc = ncsecvars['nconc'][...]
        temp = ncsecvars['temp'][...]
        w = ncsecvars['w'][...]

        ncsecfile.close()

        #compute quasi steady state ss 
        A = g*(L_v*R_a/(C_ap*R_v)*1/temp - 1)*1./R_a*1./temp
        ss = w*A/(4*np.pi*D*nconc*meanr)

        #load primary datafiles
        ncprimfile = MFDataset(DATA_DIR + model_dir + 'wrfout_d01_2014*', 'r')
        ncprimvars = ncprimfile.variables
        
        #get relevant primary variables from wrf output
        LWC = ncprimvars['QCLOUD'][...]
        SS = ncprimvars['SSW'][...]
        LAT = ncprimvars['XLAT'][...]
        LON = ncprimvars['XLONG'][...]
        
        ncprimfile.close()

        #extend latitude and longitude to all altitudes
        LAT = np.transpose(np.tile(LAT, [66,1,1,1]), [1, 0, 2, 3])
        LON = np.transpose(np.tile(LON, [66,1,1,1]), [1, 0, 2, 3])
        print(np.shape(LON))
        print(np.shape(ss))

        #make filter mask
        #mask = LWC > lwc_cutoff
        mask = np.logical_and.reduce(( \
                                    (LWC > lwc_cutoff), \
                                    (LAT < -3.1), \
                                    (LAT > -3.3), \
                                    (LON < -60.5), \
                                    (LON >  -60.8)))
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (LAT < -3.1), \
        #                            (LAT > -3.3), \
        #                            (LON < -60.5), \
        #                            (LON >  -60.8), \
        #                            (nconc > 3.e6)))
        
        #mask = np.logical_and.reduce(( \
        #                            (LWC > lwc_cutoff), \
        #                            (np.abs(w) > 1)))
        
        print(np.shape(mask))
        print('num above lwc cutoff and nonzero w: ', np.sum(mask))

        #do regression analysis
        m, b, R, sig = linregress(ss[mask]*100, SS[mask]*100)
        print(m, b, R**2)

        #count number of points outside range [-100, 100] for qss set
        n_hi = np.sum(np.logical_and.reduce(( \
                                    (ss > 1), \
                                     mask)))
        n_lo = np.sum(np.logical_and.reduce(( \
                                    (ss < -1), \
                                     mask)))
        print(model_label)
        print('n_hi: ', n_hi)
        print('n_lo: ', n_lo)

        n_q1 = np.sum(np.logical_and.reduce(( \
                                    (ss > 0), \
                                    (SS > 0), \
                                     mask)))
        n_q2 = np.sum(np.logical_and.reduce(( \
                                    (ss < 0), \
                                    (SS > 0), \
                                     mask)))
        n_q3 = np.sum(np.logical_and.reduce(( \
                                    (ss < 0), \
                                    (SS < 0), \
                                     mask)))
        n_q4 = np.sum(np.logical_and.reduce(( \
                                    (ss > 0), \
                                    (SS < 0), \
                                     mask)))

        print('Number of points in Q1:', n_q1)
        print('Number of points in Q2:', n_q2)
        print('Number of points in Q3:', n_q3)
        print('Number of points in Q4:', n_q4)
        print()

        ##get limits of the data for plotting purposes
        #xlim_max = np.max(np.array( \
        #                [np.max(SS_qss[mask]), \
        #                 np.max(SS_wrf[mask])]))
        #xlim_min = np.min(np.array( \
        #                [np.min(SS_qss[mask]), \
        #                 np.min(SS_wrf[mask])]))
        #ax_lims = np.array([100*xlim_min, 100*xlim_max])
        #print(ax_lims)
        
        ax_lims = np.array([-100, 100])
        #plot the supersaturations against each other with regression line
        fig, ax = plt.subplots()
        ax.scatter(ss[mask]*100, SS[mask]*100, c=colors['ss'])
        ax.plot(ax_lims, np.add(b, m*ax_lims), \
                        c=colors['line'], \
                        linestyle='dashed', \
                        linewidth=3, \
                        label=('m = ' + str(np.round(m, decimals=2)) + \
                                ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xlabel('Quasi steady state SS (%)')
        ax.set_ylabel('WRF SS (%)')
        fig.legend()
        fig.set_size_inches(21, 12)
        ts('End plot')
        
        ts('Start save')
        outfile = FIG_DIR + versionstr + 'subregion_qss_vs_fan_' \
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
