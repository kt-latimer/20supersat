"""
1D simulation of moist adiabatic parcel (no entrainment)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from onedimsim import BASE_DIR, DATA_DIR, FIG_DIR

versionstr = 'v57_'

#plot stuff
matplotlib.rcParams.update({'font.size': 24})
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
Mm_v = .01806 #Molecular weight of water vapour (kg/mol)
R = 8.317 #universal gas constant (J/(mol K))
R_a = R/Mm_a #Specific gas constant of dry air (J/(kg K))
R_v = R/Mm_v #Specific gas constant of water vapour (J/(kg K))
rho_l = 1000. #density of liquid water (kg/m^3)
gamma_e = -0.007 #env lapse rate (K/m)
P_e0 = 101000 #surf press (Pa)
T_e0 = 290 #surf temp (K)

def get_env_dens(z):
    rho_e = 1./(R_a*(T_e0 + gamma_e*z))*P_e0*(1 +
            gamma_e*z/T_e0)**(-1*g/(R_a*gamma_e))
    return rho_e

def get_sat_vap_pres(T):
    #sat vapor pressure
    P_vs = 611.2*np.exp(17.67*(T - 273)/(T - 273 + 243.5))
    return P_vs

#init conds and simulation constants
r0 = 5.e-6 #init radius of cloud droplets (unif distb) (m)
T0 = 281.5 #init parcel temp (K)
V0 = 1. #init parcel vol (m^3)
print(get_sat_vap_pres(T0))
m_v0 = 1100*V0/(R_v*T0)#init mass water vapor in parcel (kg)
z0 = 1000 #init parcel heigh (m)
zdot0 = 0.25#2.5 #init parcel vert vel (m/s)
m_a = (get_env_dens(z0)*R_a*(T_e0 + gamma_e*z0) \
        - m_v0*R_v*T0/V0)*V0/(R_a*T0) #mass dry air in parcel (kg)
Ndrops = 3.e4#0#3.e8 #number cloud droplets in parcel
m_w = m_v0 + 4./3.*rho_l*np.pi*r0**3.*Ndrops #mass water (gas + liq) in parcel (kg)
print(m_a, m_v0, m_w)
print((T_e0 + gamma_e*z0))
dt = 0.01 #simulation time step (s)
Nsteps = 5000 #number of time steps

def main():
    #initialize
    init_state = (m_v0, r0, T0, V0, z0, zdot0)
    time_series = [i*dt for i in range(Nsteps)]
    state_series = [init_state]

    #run through time series
    old_state = init_state
    for i in range(Nsteps-1):
        new_state = update(*old_state)
        state_series.append(new_state)
        old_state = new_state

    fig, ax = plt.subplots(1, 5)

    #plot condensation rate vs time
    q_v = np.array([state[0]/(state[0] + m_a) for state in state_series])
    dq_v = q_v[1:] - q_v[:-1]
    cond_rate = dq_v/dt
    #ax[0].plot(time_series[:-1], cond_rate, '-')
    #ax[0].set_xlabel('Time (s)')
    #ax[0].set_ylabel('dq_v/dt (s^-1)')
    zdot = np.array([state[5] for state in state_series])
    zdubdot = (zdot[1:] - zdot[:-1])/dt
    ax[0].plot(time_series[:-1], zdubdot, '-')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('zdubdot (m/s^2)')
    #plot num conc vs time
    r = np.array([state[1] for state in state_series])
    n = np.array([Ndrops/state[3] for state in state_series])
    #ax[1].plot(time_series, q_v, '-')
    #ax[1].set_xlabel('Time (s)')
    #ax[1].set_ylabel('m_v (kg)')
    #ax[1].plot(time_series, n, '-')
    #ax[1].set_xlabel('Time (s)')
    #ax[1].set_ylabel('Num. conc. (m^-3)')

    #plot ss vs time
    T = np.array([state[2] for state in state_series])
    #ax[1].plot(time_series, T, '-')
    #ax[1].set_xlabel('Time (s)')
    #ax[1].set_ylabel('T (K)')
    P_v = np.array([state[0]/state[3]*R_v*state[2] \
                    for state in state_series])
    P_vs = get_sat_vap_pres(T)
    SS = P_v/P_vs - 1
    ax[1].plot(time_series, SS*100, '-')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('SS (%)')
    #ax[2].plot(time_series, SS*100, '-')
    #ax[2].set_xlabel('Time (s)')
    #ax[2].set_ylabel('SS (%)')
    #ax[2].plot(time_series, zdot, '-')
    #ax[2].set_xlabel('Time (s)')
    #ax[2].set_ylabel('zdot (m/s)')

    ##plot parcel to env press ratio vs time
    rho_a = np.array([m_a/state[3] for state in state_series])
    P_p = P_v + rho_a*R_a*T
    #rho_e = np.array([get_env_dens(state[4]) for state in state_series])
    #T_e = T_e0 + gamma_e*np.array([state[4] for state in state_series])
    #P_e = rho_e*R_a*T_e
    #ax[3].plot(time_series, P_p/P_e, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('P_p/P_e')
    #plot z vs time
    z = np.array([state[4] for state in state_series])
    #ax[3].plot(time_series, z, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('z (m)')
    ax[3].plot(time_series, r, '-')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('r (m)')

    #plot r vs time
    #r = np.array([state[1] for state in state_series])
    #ax[3].plot(time_series, r, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('r (m)')

    w = np.array([state[5] for state in state_series])
    #ax[1].plot(time_series, w, '-')
    #ax[1].set_xlabel('Time (s)')
    #ax[1].set_ylabel('w (m/s)')
    Q1 = 1./T*(L_v*g/(R_v*C_ap*T) - g/R_a)
    Q2 = rho_a*(R_v*T/P_vs + R_a*L_v**2/(R_v*P_p*T*C_ap))
    dSS = SS[1:] - SS[:-1]
    v = Q1[:-1]*w[:-1] - Q2[:-1]*dq_v/dt
    v1 = Q1[:-1]*w[:-1]
    v2 = Q2[:-1]*dq_v/dt
    dP_v = P_v[1:] - P_v[:-1]
    dP_vs = P_vs[1:] - P_vs[:-1]
    r = np.array([state[1] for state in state_series])
    rdot = (r[1:] - r[:-1])/dt
    F_k = (L_v/(R_v*T) - 1)*(L_v*rho_l/(K*T))
    F_d = rho_l*R_v*T/(D*P_vs)
    #SSqss = Q1[:-1]*w[:-1]*(F_k[:-1] + \
    #        F_d[:-1])/(4*np.pi*rho_l*Ndrops/m_a*r[:-1])
    SSqss = Q1[:-1]*w[:-1]*(F_k[:-1] + \
            F_d[:-1])/(4*np.pi*rho_l*Ndrops/m_a*r[:-1]*Q2[:-1])
    ax[1].plot(time_series[:-1], SSqss*100, '-')

    #for i, val in enumerate(dSS):

        #print(val/dt, SS[i], w[i], r[i]*rdot[i]*(F_k[i] + F_d[i]))
        #print(val/dt, SS[i], Q1[i]*w[i], Q2[i]*dq_v[i]/dt, SS[i]*4*np.pi*rho_l*Ndrops/m_a*r[i]*Q2[i]/(F_k[i] + F_d[i]))
        #print(dq_v[i]/dt, 4*np.pi*r[i]**2.*rdot[i]*Ndrops*rho_l/m_a)
        #print(val, dP_v[i]/P_vs[i], (val + 1)*dP_vs[i]/P_vs[i], dP_v[i]/P_vs[i]
        #- (val + 1)*dP_vs[i]/P_vs[i])
    #plot T vs time
    #ax[3].plot(time_series, z, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('z (m)')
    #ax[0].plot(time_series[:-1], 100*dSS/dt, '-')
    #ax[0].set_xlabel('Time (s)')
    #ax[0].set_ylabel('dSS/dt (%/s)')
    #ax[2].plot(time_series[:-1], P_vs[:-1]*dP_v/dt, '-')
    #ax[2].set_xlabel('Time (s)')
    #ax[2].set_ylabel('P_vs*dP_v/dt (Pa^2/s)')
    #ax[3].plot(time_series[:-1], P_v[:-1]*dP_vs/dt, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('P_v*dP_vs/dt (Pa^2/s)')
    #ax[2].plot(time_series[:-1], 100*v1, '-')
    #ax[2].set_xlabel('Time (s)')
    #ax[2].set_ylabel('Q1*w (%/s)')
    #ax[3].plot(time_series[:-1], -100*v2, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('-Q2*dq_v/dt (%/s)')
    ax[2].plot(time_series, Q1*w, '-')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Q1 (%/m)')
    #ax[3].plot(time_series[:-1], -1*Q2[:-1]*dq_v/dt, '-')
    #ax[3].set_xlabel('Time (s)')
    #ax[3].set_ylabel('-Q2 (%)')

    rho_e = np.array([get_env_dens(state[4]) for state in state_series])
    rho_p = np.array([(m_a + m_w)/state[3] for state in state_series])
    #ax[4].plot(time_series, rho_e, '-')
    #ax[4].set_xlabel('Time (s)')
    #ax[4].set_ylabel('rho_e (kg/m^3)')
    #ax[4].plot(time_series, rho_p, '-')
    #ax[4].set_ylabel('rho_p (kg/m^3)')
    ax[4].plot(time_series[:-1], dq_v/dt, '-')
    ax[4].set_xlabel('Time (s)')
    ax[4].set_ylabel('dq_v/dt (s^-1)')

    fig.set_size_inches(36, 12)
    outfile = FIG_DIR + versionstr + 'onedimsim_figure.png'
    plt.savefig(outfile)
    plt.close(fig=fig)

def update(m_v, r, T, V, z, zdot):
    #print(m_v, r, T, V, z, zdot)
    rho_a = m_a/V
    rho_e = get_env_dens(z)
    rho_c = (m_w - m_v)/V
    rho_v = m_v/V
    n = Ndrops/V
    zdubdot = -1*g*(1 - rho_e/(rho_a + rho_c + rho_v))

    dz = zdot*dt
    dzdot = zdubdot*dt

    P_v = rho_v*R_v*T
    P_vs = get_sat_vap_pres(T)

    #quantities defined in ch 7 of Rogers and Yau
    F_k = (L_v/(R_v*T) - 1)*(L_v*rho_l/(K*T))
    F_d = rho_l*R_v*T/(D*P_vs)
    rdot = 1./(r*(F_d + F_k))*(P_v/P_vs - 1)
    #print(r, rdot, P_v, P_vs)

    dr = rdot*dt
    dm_v = -3.*rdot/r*(m_w - m_v)*dt
    P_e = (T_e0 + gamma_e*z)*R_a*rho_e #(R_a*rho_a + R_v*rho_v)
    dT = -1*V/(m_a*C_ap + m_v*C_vp)*(g*rho_e/(rho_a + rho_c + rho_v)*dz +
            L_v*m_a*dm_v/(m_a + m_v)**2.)
    dV = V/P_e*(P_e*dT/T + dm_v/V*R_v*T + g*rho_e*dz)
    return (m_v+dm_v, r+dr, T+dT, V+dV, z+dz, zdot+dzdot)

if __name__ == "__main__":
    main()

