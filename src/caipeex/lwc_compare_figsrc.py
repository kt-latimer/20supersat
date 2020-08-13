"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import listdir

from caipeex import BASE_DIR, DATA_DIR, FIG_DIR
from caipeex.utils import linregress

#for scatterting
colors = {'ss': '#88720A'}
versionstr = 'v1_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

lwc_filter_val = 1.e-4
w_cutoff = 2

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
    for each date and for all dates combined, create and save w histogram.
    """
    files = [f for f in listdir(DATA_DIR + 'npy_proc/')]
    used_dates = []
    i = 0
    for f in files:
        #get flight date and compare if already processed
        date = f[-12:-4]
        if date in used_dates:
            continue
        else:
            print(date)
            used_dates.append(date)
        
        #get met data for that date
        filename = DATA_DIR + 'npy_proc/MET_' + date + '.npy' 
        metdata = np.load(filename, allow_pickle=True).item()

        #get dsd data and create new file with lwc entry
        filename = DATA_DIR + 'npy_proc/DSD_' + date + '.npy'
        dataset = np.load(filename, allow_pickle=True).item()

        time = metdata['data']['sectime']#in seconds
        pres = metdata['data']['pres']
        temp = metdata['data']['temp']
        rho_air = pres/(R_a*temp) 
        print(rho_air)
        lwc_calc = metdata['data']['lwc_cdp']/rho_air #data in kg/m3
        lwc = dataset['data']['lwc_cloud'] 
        print(np.max(lwc))
        print(np.max(lwc_calc))
        m, b, R, sig = linregress(lwc, lwc_calc)
        print(m, b, R**2)

        if i == 0:
            lwc_calc_alldates = lwc_calc
            lwc_alldates = lwc
        else:
            lwc_calc_alldates = np.concatenate((lwc_calc_alldates, lwc_calc))
            lwc_alldates = np.concatenate((lwc_alldates, lwc))

        xmin = np.min(lwc)
        xmax = np.max(lwc)
        ymin = np.min(lwc_calc)
        ymax = np.max(lwc_calc)

        #make histogram
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 12)
        ax.scatter(lwc, lwc_calc)#, color=colors['tot_derv'])
        ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
                color='k', linestyle='--', \
                label=('m = ' + str(np.round(m, decimals=2)) + \
                    ', R^2 = ' + str(np.round(R**2, decimals=2))))
        ax.set_title(date)
        ax.set_xlabel('My LWC (kg/kg)')
        ax.set_ylabel('LWC from file (kg/kg)')
        ax.legend(loc=1)
        outfile = FIG_DIR + versionstr + 'lwc_compare_' \
                + date + '_figure.png'
        plt.savefig(outfile)
        plt.close(fig=fig)

        i += 1

    #make histogram
    m, b, R, sig = linregress(lwc_alldates, lwc_calc_alldates)
    print(m, b, R**2)
    xmin = np.min(lwc_alldates)
    xmax = np.max(lwc_alldates)
    print(lwc_alldates.shape)
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.scatter(lwc_alldates, lwc_calc_alldates)#, color=colors['tot_derv'])
    ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
            color='k', linestyle='--', \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.set_xlabel('My LWC (kg/kg)')
    ax.set_ylabel('LWC from file (kg/kg)')
    ax.legend(loc=1)
    outfile = FIG_DIR + versionstr + 'lwc_compare_alldates_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
