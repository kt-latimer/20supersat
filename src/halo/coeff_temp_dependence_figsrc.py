import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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

def main():
    
    temp = np.arange(275, 300, 1)
    e_s = get_sat_vap_pres(temp)
    F_d = rho_w*R_v*temp/(D*e_s) 
    F_k = (L_v/(R_v*temp) - 1)*L_v*rho_w/(K*temp)

    fig, ax = plt.subplots()
    ax.plot(temp, F_d + F_k)
    plt.show()

def get_sat_vap_pres(T):
    """
    returns saturation vapor pressure in Pa given temp in K
    """
    e_s = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return e_s

if __name__=="__main__":
    main()
