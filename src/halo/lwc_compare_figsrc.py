"""
Plot vertical wind velocity distribution from ADLR measurements, by date and in
aggregate.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo import BASE_DIR, DATA_DIR, FIG_DIR
from halo.utils import get_datablock, get_ind_bounds, \
                        match_multiple_arrays, high_bin_cas, \
                        pad_lwc_arrays, linregress

#for plotting
#colors = {'control': '#777777', 'modified': '#88720A'}
versionstr = 'v2_'

matplotlib.rcParams.update({'font.size': 21})
matplotlib.rcParams.update({'font.family': 'serif'})

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

    dates = ['20140906', '20140909', '20140911', '20140912', '20140916', \
         '20140919', '20140918', '20140921', '20140927', '20140928', \
         '20140930', '20141001']
    
    for i, date in enumerate(dates):
        print(date)
        #load data
        adlrfile = DATA_DIR + 'npy_proc/ADLR_' + date + '.npy'
        adlrdata = np.load(adlrfile, allow_pickle=True).item()
        casfile = DATA_DIR + 'npy_proc/CAS_' + date + '.npy'
        casdata = np.load(casfile, allow_pickle=True).item()
        cdpfile = DATA_DIR + 'npy_proc/CDP_' + date + '.npy'
        cdpdata = np.load(cdpfile, allow_pickle=True).item()

        #pad lwc arrays with nan values (TODO: correct data files permanently
        #and remove this section of the code)
        casdata = pad_lwc_arrays(casdata, True, True)
        cdpdata = pad_lwc_arrays(cdpdata, True, True)

        #loop through reasonable time offset range ($\pm$ 9 sec)
        [adlrinds, casinds, cdpinds] = match_multiple_arrays(
            [np.around(adlrdata['data']['time']), \
            np.around(casdata['data']['time']), \
            np.around(cdpdata['data']['time'])])
        datablock = get_datablock(adlrinds, casinds, cdpinds, \
                                    adlrdata, casdata, cdpdata)
        lwc_calc = casdata['data']['lwc_calc'][casinds]

        #remove rows with error values (except vert wind vel cause it's shit)
        goodrows = []
        for j, row in enumerate(datablock):
            if sum(np.isnan(row)) == 0:
                goodrows.append(j)
        datablock = datablock[goodrows, :]
        pres = datablock[:, -2]
        temp = datablock[:, 1]
        rho_air = pres/(R_a*temp)
        print(rho_air)
        #need to correct mistake in halo_data_polish scaling factor
        #and change to kg/kg
        lwc_calc = lwc_calc[goodrows]/rho_air*1.e-6


        lwc = datablock[:, high_bin_cas+0]

        if i == 0:
            lwc_alldates = lwc
            lwc_calc_alldates = lwc_calc
        else:
            lwc_alldates = np.concatenate((lwc_alldates, lwc))
            lwc_calc_alldates = \
                        np.concatenate((lwc_calc_alldates, lwc_calc))

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
