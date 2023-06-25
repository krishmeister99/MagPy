#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:15:52 2021

@author: krishnanganesh and harveymarples
"""

import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import skyrmpy_functions as sk
from scipy.integrate import ode
from scipy import constants
# from scipy.fft import fft , fftfreq
from matplotlib import animation

# =============================================================================
# Unit scales:
# =============================================================================

# Energy scale:
J_exch = 1e-3 #units: eV

# Time scale:
T = (constants.hbar / (J_exch * 1.6e-19))*(1e12) #units:pico-seconds

# Length scale:
a = 5 # units: Å

# Velocity scale:
V = (a*1e-10)/(T*1e-12) # units: m/s

#=============================================================================
# CLASS SETUP
#=============================================================================

# Skyrmion class/object 
class SystemParams:
    def __init__(self, J, Z,K, D, omega, array_size, alpha, beta, B, j_c, L):
        self.J = J
        self.Z = Z
        self.K = K
        self.D = D
        self.array_size = array_size
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.B = B
        self.j_c = j_c
        self.L = L
        self.delta_x = L/array_size

# Create system parameters object
system_params = SystemParams(J = 1, # Exchange Stiffness
                             Z = 0.8, # Zeeman Coupling Strength
                             K = 0, # Uniaxial Anisotropy Strength
                             D = 1, # Dzyaloshinskii-Moriya Interaction
                             omega = 3 , # Frequency of external magnetic field.
                             array_size = 51,
                             
                             alpha = 0.04, # Gilbert-Damping Constant
                             beta = 0.01, # Non-adiabaticity for STT
                             B = np.array([0,0,1]),
                             #B = np.array([0, 0, -1]), # External magnetic field unit vector
                             j_c = np.array([0, 0]), # Electron Drift Velocity
                             
                             L = 50 # System Length (in units of 'a')
                             )


# =============================================================================
# Skyrmion initial condition:
# =============================================================================

def phi(X,Y):
    # Function that calculates phi for magnetisation texture
    azimuth = np.arctan2(Y,X)
    return azimuth

#=============================================================================

# Create arrays of co-ordinates
x_coord = np.linspace(-system_params.L/2, system_params.L/2, system_params.array_size)
y_coord = np.linspace(-system_params.L/2, system_params.L/2, system_params.array_size)
X,Y = np.meshgrid(x_coord,y_coord)

# Theta = sk.Theta_calc(system_params)
# m = 1
# g = -np.pi/2
# Calculate magnetisation components
# m_x = np.sin(Theta) * np.cos(m*phi(X,Y)+g)
# m_y = np.sin(Theta) * np.sin(m*phi(X,Y)+g)    
# m_z = np.cos(Theta)

m_x = np.sin(X) * np.cos(Y)
m_y = np.sin(X) * np.sin(Y)    
m_z = np.cos(X)

# Landau Lifshitz Gilbert equation:
def dM_dt(t,M):
    '''
    Implements LLG equation. Contains spins precession term (HxM) and Gilbert
    damping (Mx(HxM)). EXTRA TERMS TO BE ADDED.
    Parameters
    ----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]
        
    Returns
    -------
    M_dot : Time derivative 
        
    '''
    H_cross = sk.H_matrix_coo(M, system_params, t)
    xprod1 = H_cross.dot(M)
    M_cross = sk.M_matrix_coo(M)
    xprod2 = M_cross.dot(xprod1)

    #LLG equation:
    M_dot = (1/(1+ system_params.alpha**2))*(xprod1 + system_params.alpha*(xprod2)) 
    return M_dot


#Initial condition:
M0 = sk.M_comp(m_x , m_y , m_z)

# Integrating the equation:
r = ode(dM_dt).set_integrator('vode')
r.set_initial_value(M0 , 0)
tf = 10
dt = 0.05
i = 0
Data = np.zeros(shape=( int(3*(system_params.array_size**2)) , int(tf/dt)))

# Data for plotting top_charge and velocity
time_data = np.linspace(0 , tf , int(tf/dt))
# dist_data = np.zeros(len(time_data))
energy_data = np.zeros(len(time_data))
top_charge_data = np.zeros(len(time_data))
chi_data_xx = np.zeros(len(time_data))
chi_data_yy = np.zeros(len(time_data))
chi_data_zz = np.zeros(len(time_data))# this is for steady state detection
# radius_data = np.zeros(len(time_data))

relax = False
relax_array = np.zeros(len(time_data))
diffs = np.zeros(len(time_data))
pulse_time = 0
stop_condition = '%.'+str(len(str(tf)) + 2)+'g'
while r.successful() and float(stop_condition % r.t) < tf:
    r.integrate(r.t+dt)
    Data[:,i] = r.y
    # Q_charge = sk.top_charge(r.y)
    # # # Data for plotting
    # top_charge_data[i] = Q_charge
    # energy_data[i] = np.sum(sk.tot_energy(r.y , system_params))
    # # # #Chi data:
    # chi = sk.sus(r.y)
    # chi_data_xx[i] = chi[0]
    # chi_data_yy[i] = chi[1]
    # chi_data_zz[i] = chi[2]# this is for steady state detection
    # # Calculating the difference between current value of chi and previous value:
    # diffs[i] = np.abs(chi_data_zz[i] - chi_data_zz[i-1])
    # system_params.B = np.array([0 , 0 , 1])
    # # Mean of the last 600 values in diff
    # if np.mean(diffs[i-600:i]) <= 1e-7 and relax == False:
    #     relax_array[i] = 1
    #     system_params.B = ((1/system_params.Z)*0.5*np.array([10 , 0 , 0 ])) + np.array([0 , 0 , 1])
    #     pulse_time = time_data[i]
    #     pulse_index = i
    #     relax = True 
    
    
    completion = 'Frame'+str(i+1)+'complete' + ' Mag field:' + str(system_params.B)
    print(completion)
                
    # Radius data:
    # radius_data[i] = sk.skyr_radius(r.y, system_params)
    
    
    i+=1


# =============================================================================
# # Animating the vector field:
# =============================================================================
title = '{} , {} , {} , {} '.format('J='+ str(system_params.J) , 'K='+str(system_params.K) , 'Z='+ str(system_params.Z) , 'D='+str(system_params.D) )
caption = '{} , {}'.format(r'$\alpha = $'+str(system_params.alpha) , r'$\beta =$'+str(system_params.beta))
fig = plt.figure(1)
ax = plt.axes(xlim=(-system_params.L/2, system_params.L/2), ylim=(-system_params.L/2, system_params.L/2))
ax.set_title(title , pad = 20)
ax.set_xlabel(caption ,labelpad = 20)
plt.gca().set_aspect('equal', adjustable='box')
cmap = 'Spectral'
time = str(time_data[0]) 
time_text = ax.text(-system_params.L/2 +1 , -system_params.L/2 +1 , time , color = 'white' )

# Initial Points on Locus for Skyrmion Boundary
# locus_points = sk.locus_calc(sk.M_exp(M0)[2], system_params)

# Add Quiver plot and Colourbar for initial condition
I = plt.imshow(m_z, origin='lower', extent = [np.min(X),np.max(X),np.min(Y),np.max(Y)], cmap=cmap)
Q = plt.quiver(X, Y, m_x, m_y, color = 'black', scale = 20)
#boundary, = ax.plot(locus_points[:,0], locus_points[:,1], color='black')
cbar = fig.colorbar(I,extend='max')
cbar.set_label('$m_{z}$')

# COMMENTED OUT THE LOCUS PLOTTER VERSION
# def init():
#       boundary.set_data([], [])
#       return boundary,

# def update_stuff(frame, Q, I, boundary, X, Y, time_text):
#     m_x = sk.M_exp(Data[:,frame])[0]
#     m_y = sk.M_exp(Data[:,frame])[1]
#     Q.set_UVC(m_x , m_y)
    
#     m_z = sk.M_exp(Data[:,frame])[2]
#     I.set_array(m_z)
    
#     time_text.set_text(str(np.around(time_data[frame] , 3)) + 'ps')
    
#     locus_points = sk.locus_calc(m_z, system_params)
#     boundary.set_data(locus_points[:,0], locus_points[:,1])
    
#     return Q, I, time_text,


# anim = animation.FuncAnimation(fig, update_stuff, init_func = init, frames = np.int(tf/dt),  fargs=(Q, I, boundary, X, Y, time_text),
#                                 interval=50, blit=False, repeat_delay = 1500)


def update_stuff(frame, Q, I, X, Y, time_text):
    m_x = sk.M_exp(Data[:,frame])[0]
    m_y = sk.M_exp(Data[:,frame])[1]
    Q.set_UVC(m_x , m_y)
    
    m_z = sk.M_exp(Data[:,frame])[2]
    I.set_array(m_z)
    
    time_text.set_text(str(np.around(time_data[frame] , 3)) + 'ps')
    
    return Q, I, time_text,


anim = animation.FuncAnimation(fig, update_stuff, frames = int(tf/dt),  fargs=(Q, I, X, Y, time_text),
                                interval=0, blit=False, repeat_delay = 1500)


plt.show()

# =============================================================================
# ##### EXTRA PLOTS ######
# =============================================================================
#Plotting topological charge as a function of time:
fig2 = plt.figure(2)
plt.plot(time_data, top_charge_data)
plt.xlabel('Time/ps')
plt.ylabel('Topological Charge $\mathcal{Q}$')
plt.xlim([0 , tf])
plt.ylim([-2 , 2])


# # Plotting skyrmion speed as a function of time:
# fig3 = plt.figure(3)
# plt.plot(time_data,velocity_data)
# plt.xlabel('Time/ps')
# plt.ylabel('Skyrmion Velocity Magnitude/$ms^{-1}$')


# Plot distortion parameter
# fig4 = plt.figure(4)
# plt.plot(time_data,dist_data)
# plt.xlabel('Time/ps')
# plt.ylabel('Distortion Parameter $\delta$')

# #Plotting suscpetibility:
# fig5 = plt.figure(5)
# plt.plot(time_data , chi_data)
# plt.xlabel('Time')
# plt.ylabel(r'$\chi(t)$')
# plt.axvline(x = pulse_time , color = 'r' , linestyle = '--')

# # Fourier transform of susceptibility:
# N = int(tf/dt)
# fig6 = plt.figure(6)    
# chi_w = fft(chi_data)
# freq = fftfreq(N , dt)[:N//2]
# plt.plot(freq, np.abs(chi_w.imag[:N//2]))
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$Im[\chi_{zz}(\omega)]$')
# plt.ylim(bottom = 0)

# Plotting radius against time
# fig7 = plt.figure(7)
# plt.plot(time_data, radius_data)
# plt.xlabel('Time')
# plt.ylabel('Radius')
# plt.axvline(x = pulse_time , color = 'r' , linestyle = '--')

# # plotting truncated signal:
# chi_trunc = chi_data[pulse_index:]
# # chi_trunc = signal.detrend(chi_trunc, type == 'linear')
# time_trunc = time_data[pulse_index:]- time_data[pulse_index]
# fig8  = plt.figure(8)
# plt.plot(time_trunc , chi_trunc)
# plt.xlabel(r'$t$')
# plt.ylabel(r'$\chi(t)$')
# plt.title('Truncated signal')
# plt.xlim(left = 0)
# plt.grid()

# # fourier transform of truncated signal:
# N = len(time_trunc)
# fig9 = plt.figure(9)
# chi_trunc_fft = fft(chi_trunc)
# freq_trunc = fftfreq(N , dt)[:N//2]
# plt.plot(freq_trunc , (2/N)*np.abs(chi_trunc_fft.imag[:N//2]))
# plt.xlim([0 , 1])
# plt.xlabel(r'$\omega$')
# plt.ylabel(r'$Im[\chi_{zz}(\omega)]$')
# plt.title('Fourier spectrum for truncated signal')
# plt.grid()

fig10 = plt.figure(10)
plt.plot(time_data , energy_data)
plt.xlabel(r'$t$')
plt.ylabel(r'$E_{tot}$')

##############################################################################
# AUTOMATICALLY SAVE IMPORTANT DATA
##############################################################################

# Create new folder
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H-%M-%S")
folder_name = 'Data_Folder_' + dt_string
os.mkdir(folder_name)
current_dir = os.getcwd()

# Save system params and others in text file
text_file = open(current_dir + '/' + folder_name + '/' + 'READ_ME.txt', 'w')
text_file.write("J = " + str(system_params.J) + "     Exchange Stiffness")
text_file.write('\n')
text_file.write("Z = " + str(system_params.Z) + "     Zeeman Coupling Strength")
text_file.write('\n')
text_file.write("K = " + str(system_params.K) + "     Uniaxial Anisotropy Strength")
text_file.write('\n')
text_file.write("D = " + str(system_params.D) + "     DM Interaction Strength")
text_file.write('\n')
text_file.write("L = " + str(system_params.L) + "     System size")
text_file.write('\n')
text_file.write("array_size = " + str(system_params.array_size) + "     Array Size")
text_file.write('\n')
text_file.write("alpha = " + str(system_params.alpha) + "     Gilbert Damping Constant")
text_file.write('\n')
text_file.write("beta = " + str(system_params.beta) + "     Non-adiabaticity for STT")
text_file.write('\n')
text_file.write("tf = " + str(tf) + "     Integration time")
text_file.write('\n')
text_file.write("dt = " + str(dt) + "     Integration time-step")
text_file.write('\n')
text_file.close()

# # Save pulse index as text file
# if relax == True:
#     text_file2 = open(current_dir + '/' + folder_name + '/' + 'pulse_index.txt', 'w')
#     text_file2.write(str(pulse_index))
#     text_file2.close()

# # Save arrays
np.save(current_dir + '/' + folder_name + '/' + 'Data.npy', Data)
# np.save(current_dir + '/' + folder_name + '/'+'Q_charge.npy', Q_charge)
# np.save(current_dir + '/' + folder_name + '/'+'energy_data.npy', energy_data)
# np.save(current_dir + '/' + folder_name + '/'+'chi_data_xx.npy', chi_data_xx)
# np.save(current_dir + '/' + folder_name + '/'+'chi_data_yy.npy', chi_data_yy)
# np.save(current_dir + '/' + folder_name + '/'+ 'chi_data_zz.npy', chi_data_zz)
# np.save(current_dir + '/' + folder_name + '/'+ 'time_data.npy', time_data)
# np.save(current_dir + '/' + folder_name + '/'+ 'top_charge_data.npy', top_charge_data)

