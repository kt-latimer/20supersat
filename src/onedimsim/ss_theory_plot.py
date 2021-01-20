import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#plot stuff
#matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'font.family': 'serif'})
colors = {'line': '#000000', 'ss': '#88720A'}

#physical constants
C_ap = 1005. #dry air heat cap at const P (J/(kg K))
C_av = 718. #dry air heat cap at const V (J/(kg K))
C_vp = 1424. #dry air heat cap at const P (J/(kg K))
C_vv = 1890. #dry air heat cap at const V (J/(kg K))
D = 0.23e-4 #diffus coeff water in air (m^2/s)
g = 9.8 #grav accel (m/s^2)
K = 2.4e-2 #therm conductivity of air (J/(m s K))
L_v = 2501000. #latent heat of evaporation of water (J/kg)
Mm_a = .02896 #Molecular weight of dry air (kg/mol)
Mm_s = .058 #Molecular weight of salt (kg/mol)
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of liquid water (kg/m^3)
rho_s = 2.16e3 #density of salt (kg/m^3) 
sigma_w = 70e-3 #surface tension bt water and air (N/m)
theta_s = 1. #chem potential for salt (1/mol)
v = 2. #valency of salt

r = np.logspace(-4.8, -4)
S = 0.9 
T = 280

r_d_vals = np.logspace(-8, -5, num=7) 
m_s_vals = 4./3.*np.pi*r_d_vals**3.*rho_s

def main():

    fig, ax = plt.subplots()

    for i, r_d in enumerate(r_d_vals):
        m_s = m_s_vals[i]
        #r = np.linspace(r_d*1.01, 1.e-4)
        r = np.linspace(1.e-5, 1.e-4)
        y = 1./r*(S - np.exp(2*sigma_w/(R_v*rho_l*T*r) - \
        (3*v*theta_s*m_s*Mm_v)/((4*np.pi*rho_l*Mm_s)*(r**3. - r_d**3.))))
        ypr = 1./r*(S - 1)
        #ax.plot(r, np.log10(y), label=str(r_d))
        ax.plot(r, y, label=str(r_d))
        ax.plot(r, ypr)

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
