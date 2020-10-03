"""
check my calculated lwc against values in the files
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

from revcaipeex import DATA_DIR, FIG_DIR
from revcaipeex.ss_qss_calculations import get_lwc, linregress

#for plotting
versionstr = 'v1_'
matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

##
## physical constants
##
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

lwc_filter_val = 1.e-4
w_cutoff = 2

cutoff_bins = False 

def main():

    with open('good_dates.txt', 'r') as readFile:
        good_dates = [line.strip() for line in readFile.readlines()]

    caipeex_lwc_alldates = None
    my_lwc_alldates = None

    for date in good_dates:
        metfile = DATA_DIR + 'npy_proc/MET_' + date + '.npy'
        met_dict = np.load(metfile, allow_pickle=True).item()
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdp_dict = np.load(cdpfile, allow_pickle=True).item()

        temp = met_dict['data']['temp']
        pres = met_dict['data']['pres']
        rho_air = pres/(R_a*temp)

        caipeex_lwc = met_dict['data']['lwc_cdp']/rho_air
        my_lwc = get_lwc(dsd_dict,cutoff_bins)

        m, b, R, sig = linregress(caipeex_lwc, my_lwc)

        print(date)

        caipeex_lwc_alldates = add_to_alldates_array(caipeex_lwc, caipeex_lwc_alldates)
        my_lwc_alldates = add_to_alldates_array(my_lwc, my_lwc_alldates)

        make_and_save_lwc_compare(caipeex_lwc, my_lwc, date, \
                                    versionstr, cutoff_bins, m, b, R)

    m, b, R, sig = linregress(caipeex_lwc_alldates, my_lwc_alldates)

    make_and_save_lwc_compare(caipeex_lwc_alldates, my_lwc_alldates, \
                                'alldates', versionstr, cutoff_bins, m, b, R)

def add_to_alldates_array(lwc, lwc_alldates):

    if lwc_alldates is None:
        return lwc
    else:
        return np.concatenate((lwc_alldates, lwc))

def make_and_save_lwc_compare(caipeex_lwc, my_lwc, label, \
                                versionstr, cutoff_bins, m, b, R):

    xmin = np.min(caipeex_lwc)
    xmax = np.max(caipeex_lwc)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(21, 12)
    ax.scatter(caipeex_lwc, my_lwc)
    ax.plot([xmin, xmax], [m*xmin + b, m*xmax + b], \
            color='k', linestyle='--', \
            label=('m = ' + str(np.round(m, decimals=2)) + \
                ', R^2 = ' + str(np.round(R**2, decimals=2))))
    ax.plot([xmin, xmax], [xmin, xmax], \
            color='r', linestyle='--', \
            label='1:1')
    ax.legend()
    ax.set_xlabel('CAIPEEX LWC (kg/kg)')
    ax.set_ylabel('My LWC (kg/kg)')
    ax.set_title(label + ' LWC compare' \
                    + ', cutoff_bins=' + str(cutoff_bins))
    outfile = FIG_DIR + versionstr + 'check_lwc_' \
            + label + '_figure.png'

    plt.savefig(outfile)
    plt.close(fig=fig)    

if __name__ == "__main__":
    main()
