#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:23:46 2021

@author: krishnanganesh and harveymarples
"""

import numpy as np
# from numba import njit
from scipy import sparse
from scipy import optimize
# from scipy import signal
from scipy.signal import savgol_filter
from scipy import linalg
import math
#Initial condition skyrmion:
def Theta_calc(system_params):
    Z = system_params.Z
    J = system_params.J
    D = system_params.D
    array_size = system_params.array_size
    L = system_params.L
    # Calculates the value of Theta at each point on the lattice:
    
    # Coordinates:
    x_coord = np.linspace(-L/2 , L/2 , array_size)
    y_coord = np.linspace(-L/2 , L/2 , array_size)
    delta = (L) / (array_size -1) 
    # origin index :
    o = int(0.5*(array_size - 1))
    
    # Theta field:
    Theta = np.zeros(shape = (len(x_coord) , len(y_coord)))
    
    # Boundary conditions:    
    # At the origin, the spin must be pointing downards, so Theta[origin] = pi
    Theta[o , o] = np.pi
    
    # At the edges of our lattice (i.e. far away from the skyrmion centre), the spins must point up.
    # This is already the case in the Theta matrix.
    
    # Some functions to calculate the radius and azimuthal angle:
    def radius(i , j):
        deltaX = (x_coord[j] - x_coord[o])
        deltaY = (y_coord[i] - y_coord[o])
        return np.sqrt(deltaX**2 + deltaY**2)
    
    def phi(i, j):
        deltaX = (x_coord[j] - x_coord[o])
        deltaY = (y_coord[i] - y_coord[o])
        return np.arctan2(deltaY , deltaX)
    
    # Dimensionless parameter:
    Z_eff = (Z*J)/(D**2)
    
    # We must sweep over all lattice sites that are NOT included in the boundary conditions.
    # This means I must exclude the central point {i = 10 , j = 10} and the edges from the for loops I use:
    
    # I'll try using an array that contains a list of the allowed values for i and j:
    sites = np.arange(1 , array_size - 1 , 1)
    delta_t = 0.01
    not_converge = True
    k = 0
    w = 1.9
    
    while not_converge:
        Theta_last = Theta.copy()
        for i in sites:
            for j in sites:
                # # If both i = 10 and j = 10, we skip to the next iteration:
                if i == o and j == o:
                    continue
                cos = np.cos(phi(i , j))
                sin = np.sin(phi(i , j))
                r = radius(i , j)
                #Derivative terms:
                deriv1 = 0.5*(1/delta)*(cos*(Theta[i , j+1] - Theta[i , j-1]) + \
                                        sin*(Theta[i+1 , j] - Theta[i -1 , j]))
                deriv2 = (1/delta**2)*((cos**2)*(Theta[i , j+1] + Theta[i , j-1]) + (sin**2)*(Theta[i+1 , j] + Theta[i-1 , j]) + \
                    0.25*np.sin(2*phi(i , j))*(Theta[i+1 , j+1] + Theta[i-1 , j-1] - Theta[i+1 , j-1] - Theta[i-1 , j+1]) - \
                        2*Theta[i , j])
                # Non-derivative terms:
                non = 2*(np.sin(Theta[i,j]))**2 - 0.5*np.sin(2*Theta[i , j])*(1/r) - Z_eff*r*np.sin(Theta[i , j]) 
                
                #Shift to Theta{i , j}:
                Theta[i , j] += delta_t*w*(non + deriv1 + r*deriv2)
                
    
                # testing with the laplace equation for electostatics for a sanity check:
                # deriv = (1/delta**2)*( Theta[ i , j+1] + Theta[ i , j-1] + Theta[i+1 , j] + Theta[i-1 , j] - 4*Theta[i , j])
                # Theta[i , j] += delta_t*deriv
        if np.max(np.abs(Theta - Theta_last)) <= 1e-6:
            print("Solution converged in "+ str(k) + " steps")
            not_converge = False
            break
        k += 1
    return Theta   
# =============================================================================
# Landau-Lifshitz-Gilbert functions:
# =============================================================================
# Magnetisation texture components -> Single column vector 'M':
# @njit    
def M_comp(mx , my , mz):
    '''
    Compiles the magnetisation texture fields mx, my and mz into a 1x3N^2  
    dimensional column vector 'M' of form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]
    where N^2 is the number of lattice sites.
    Parameters
    ----------
    mx : x-component of magnetisation at each lattice site
    my : y-component of magnetisation at each lattice site
    mz : z-component of magnetisation at each lattice site

    Returns
    -------
    M
    '''
    N = len(mx)
    M = np.zeros(3*(N**2) , dtype = np.float64)
    n = 0
    for i in range(N):
        for j in range(N):
            M[n] = mx[i , j]
            M[n + 1] = my[i , j]
            M[n + 2] = mz[i , j]
            n += 3
    return M

# Single column vector 'M' -> Magnetisation texture components:
# @njit
def M_exp(M):
    '''
    Expands the single column vector storing the components of all of the spins,
    into NxN matrices for each vector component

    Parameters
    ----------
    M : Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]

    Returns
    -------
    mx : x-component of magnetisation at each lattice site
    my : y-component of magnetisation at each lattice site
    mz : z-component of magnetisation at each lattice site

    '''
    N = np.int(np.sqrt(len(M)/3))
    mx = np.zeros(shape = (N , N) , dtype = np.float64)
    my = np.zeros(shape = (N , N) , dtype = np.float64)
    mz = np.zeros(shape = (N , N) , dtype = np.float64)
    for i in range(N):
        for j in range(N):
            mx[i , j] = M[3*(N*i + j)]
            my[i , j] = M[3*(N*i + j)+ 1]
            mz[i , j] = M[3*(N*i + j)+ 2]
    return mx , my , mz

# Total magnetic field:
# @njit
def H_eff(M , i , j , system_params, t):
    '''
    Takes a lattice site [i , j] and spits out the total magnetic field vector.

    Parameters
    ----------
    M : Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]
    i: x coordinate of lattice site
    j: y coordinate of lattice site
    J: Exchange energy
    Z: Zeeman coupling energy
    K: Uniaxial anisotropy
    B: External magnetic field unit vector
    D: Dzyaloshinskii-Moriya interaction strength
    t: time
    relax: a list that indicates whether the skyrmion has relaxed into it's equilibrium shape

    Returns
    -------
    H_eff : Total magnetic field vector at lattice site [i , j]

    '''
    # Get parameters from system_params object
    J = system_params.J
    Z = system_params.Z
    K = system_params.K
    B = system_params.B
    D = system_params.D
    # delta_x = system_params.delta_x
    
    N = np.int(np.sqrt(len(M)/3))
    sum_ij = 3*(N*i + j)
    sum_down = 3*(N*((i-1)%N) + j) #<-modulo for periodic boundary conditions.
    sum_up = 3*(N*((i+1)%N) + j)
    sum_right = 3*(N*i + (j+1)%N)
    sum_left = 3*(N*i + (j-1)%N)
    m_ij = np.array([ M[sum_ij] , M[sum_ij +1] , M[sum_ij +2] ])
    m_left = np.array([ M[sum_left] , M[sum_left + 1 ] , M[sum_left +2] ])
    m_right = np.array([ M[sum_right] , M[sum_right +1] , M[sum_right +2] ])
    m_up = np.array([ M[sum_up] , M[sum_up +1] , M[sum_up +2] ])
    m_down = np.array([ M[sum_down] , M[sum_down + 1] , M[sum_down + 2] ])
    # Exchange term:
    exch = J*(m_left + m_right + m_down + m_up)
    # Zeeman term:
    #Dirac delta:
    zee = Z*B
    # Uniaxial anisotropy:
    aniso = K*np.dot(m_ij , np.array([0 , 0 , 1])) * np.array([0 , 0 , 1])
    
    # Dzyaloshinskii-Moriya interaction:
    
    dmi_x = M[sum_up + 2] - M[sum_down +2]
    dmi_y = -1*(M[sum_right + 2] - M[sum_left + 2] )
    dmi_z = M[sum_right +1] - M[sum_left + 1] -M[sum_up] + M[sum_down]
    
    # UNCOMMENT THESE AT YOUR OWN RISK, LEST YOU UNLEASH THE... ANTI-SKYRMION:
    # dmi_x = M[sum_right + 2] - M[sum_left + 2]
    # dmi_y = M[sum_down + 2] - M[sum_up + 2]
    # dmi_z = (M[sum_left] - M[sum_right]) + (M[sum_up + 1] - M[sum_down + 1])
    
    DMI = (D) * np.array([dmi_x , dmi_y , dmi_z])
    H_eff = exch + zee + aniso + DMI
    return H_eff

# Total energy:
def tot_energy(M, system_params):
    J = system_params.J
    Z = system_params.Z
    K = system_params.K
    B = system_params.B
    D = system_params.D
    N =  N = np.int(np.sqrt(len(M)/3))
    H_exch = 0
    H_DMI = 0
    H_Zee = 0
    H_Ani = 0
    for k in range(N**2):
        i = int(np.floor(k/N))
        j = int(k%N)
        ij = 3*(N*i + j)
        up = 3*(N*((i+1)%N) + j)
        right = 3*(N*i + (j+1)%N)
        m_ij = np.array([ M[ij] , M[ij + 1] , M[ij + 2]])
        m_up = np.array([ M[up] , M[up + 1] , M[up + 2]])
        m_right = np.array([ M[right] , M[right + 1] , M[right + 2]])
        H_exch += -J * ( np.dot(m_ij , m_up + m_right) )
        H_DMI += D * (np.dot(np.array([1 , 0 , 0 ]) , np.cross(m_ij , m_right)) 
                      + np.dot(np.array([0 , 1 , 0 ]) , np.cross(m_ij , m_up)))
        H_Zee +=  Z * (1 - np.dot(B, m_ij))
        H_Ani += -K * (np.dot(np.array([0 , 0 , 1]) , m_ij))**2
    E = np.array([H_exch , H_DMI , H_Zee , H_Ani])
    return E
        

# Total field cross-product matrix:
# @njit
def H_matrix(M, system_params, t):
    '''
    Takes the column vector M, and spits out a matrix that takes the 
    cross product HxM at all lattice sites.

    Parameters
    ----------
    M : Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]

    Returns
    -------
    H_sp : A sparse matrix that takes the cross of product HxM f
    or all lattice sites

    '''
    
    N = np.int(np.sqrt(len(M)/3))
    H = np.zeros(shape = (3*(N**2) , 3*(N**2)) , dtype = np.float64)
    n = 0
    for i in range(N):
        for j in range(N):
            # Local demagnetisation field vector:
            H_ij = H_eff(M, i , j, system_params, t)
            # Turning vector into a cross product matrix:
            H[n , n+1] = -1*H_ij[2]
            H[n , n+2] = H_ij[1]
            H[n+1 , n+2] = -1*H_ij[0]
        
            H[n+1 , n] = -1* H[n , n+1]
            H[n+2 , n] = -1* H[n , n+2]
            H[n+2 , n+1] = -1* H[n+1 , n+2]
            n += 3 
    H_sp = sparse.csr_matrix(H)
    return H_sp

# Magnetisation cross product matrix:
# @njit
def M_matrix(M):
    '''
    Takes the column vector M, and spits out a matrix that takes the 
    cross product Mx
    Parameters
    ----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]

    Returns
    -------
    Mbar_sp : A sparse matrix that takes the cross of product Mx for 
    all lattice sites
    '''
    
    N = np.int(np.sqrt(len(M)/3))
    Mbar = np.zeros(shape = (3*(N**2) , 3*(N**2)), dtype=np.float64)
    n = 0
    for i in range(N):
        for j in range(N):
            sum_ij = 3*(N*i + j)
            Mbar[n , n+1] = -1*M[sum_ij +2]
            Mbar[n , n+2] = M[sum_ij + 1]
            Mbar[n+1 , n+2] = -1*M[sum_ij]
            
            Mbar[n+1 , n] = -1* Mbar[n , n+1]
            Mbar[n+2 , n] = -1* Mbar[n , n+2]
            Mbar[n+2 , n+1] = -1* Mbar[n+1 , n+2]
            n += 3  
    Mbar_sp = sparse.csr_matrix(Mbar)
    return Mbar_sp
# @njit
def J_STT(system_params):
    '''
    Generates a matrix that performs the STT operation (j.∇) on the magnetisation
    texture, where 'j' is the number current density vector.

    Parameters
    ----------
    N : Lattice width (in unit cells).
    j_x : x-component of the number current density vector.
    j_y : y-component of the number current density vector.

    Returns
    -------
    J_matrix : Sparse matrix performs (j.∇) operation on every magnetisation vector in the 
    lattice.

    '''
    # Get variables needed from system_params object
    j_x = system_params.j_c[0]
    j_y = system_params.j_c[1]
    N = system_params.array_size
    delta_x = system_params.delta_x
    
    
    J = np.zeros(shape = (3*N**2 , 3*N**2) , dtype = np.float64)
    for i in range(N):
        for j in range(N):
            ij = 3*(N*i + j)
            down = 3*(N*((i-1)%N) + j) #<-modulo for periodic boundary conditions.
            up = 3*(N*((i+1)%N) + j)
            right = 3*(N*i + (j+1)%N)
            left = 3*(N*i + (j-1)%N)
            # X-components of magnetisation:
            J[ij , right] = j_x*1*0.5
            J[ij , up] = j_y*1*0.5
            J[ij ,left] = -1*j_x*0.5
            J[ij , down] = -1*j_y*0.5
            # Y-components of magnetisation:
            J[ij+1 , right+1] = 1*j_x*0.5
            J[ij+1 , up+1] = 1*j_y*0.5
            J[ij+1 ,left+1] = -1*j_x*0.5
            J[ij+1 , down+1] = -1*j_y*0.5
            # Z-components of magnetisation:
            J[ij+2 , right+2] = 1*j_x*0.5
            J[ij+2 , up+2] = 1*j_y*0.5
            J[ij+2 ,left+2] = -1*j_x*0.5
            J[ij+2 , down+2] = -1*j_y*0.5
    J/=delta_x
    J_matrix = sparse.csr_matrix(J)
    return J_matrix

# Landau Lifshitz Gilbert equation:
# @njit
def dM_dt(t,M,gamma,alpha):
    '''
    Implements LLG equation. Contains spins precession term (HxM) and Gilbert
    damping (Mx(HxM)). EXTRA TERMS TO BE ADDED.
    Parameters
    ----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]
    gamma: Gyromagnetic ratio
    alpha: Gilbert damping constant
        
    Returns
    -------
    M_dot : Time derivative 
        
    '''
    H_cross = H_matrix(M)
    xprod1 = H_cross.dot(M)
    M_cross = M_matrix(M)
    xprod2 = M_cross.dot(xprod1)
    M_dot = (gamma/(1+alpha**2))*(xprod1 + alpha*(xprod2))
    return M_dot

# =============================================================================
# Locus calculator
# =============================================================================
def locus_calc(m_z, system_params):
    '''
    Calculates the co-ordinates of points on the skrmion boundary defined
    as the locus of points where m_z = 0.
    
    Parameters
    ----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]
    array_size: The number of array elements in each dimension
    x_coord: 1D Array of the x-coordinates of the system
    y_coord: 1D Array of the y-coordinates of the system
    
    Returns
    -------
    locus_points_array: Array containing the (x,y) co-ordinates of points on
    the skyrmion boundary ordered by ascending phi-values.
    
    '''
    array_size = system_params.array_size
    L = system_params.L
    
    x_coord = np.linspace(-L/2,L/2,array_size)
    y_coord = np.linspace(-L/2,L/2,array_size)
    
    # Extracting m_z values:
    #m_z = M_exp(M)[2]
    
    # Create empty array for locus points
    locus_points_x = []
    locus_points_y = []
    
    # Locus point finding loop
    for skyr_slice in range(0,array_size):
        
        # Curve fitting to find function of slice
        poly_order = 15
        fitted_params_y = np.polyfit(x_coord, m_z[skyr_slice,:], poly_order)
        fitted_params_x = np.polyfit(y_coord, m_z[:,skyr_slice], poly_order)
        
        # Create function from fitted_parameters
        f_y = np.poly1d(fitted_params_y)
        f_x = np.poly1d(fitted_params_x)
        
        # Newton-Raphson Method if slice goes below m_z=0 for y slices
        if min(m_z[skyr_slice,:]) < 0:
            
            # Find index of closest value to zero and assign NR guess
            guess_indx = np.where(min(abs(m_z[skyr_slice,:])) == abs(m_z[skyr_slice,:]))
            guess = guess_indx[0][0] * (20/array_size) - 10
            
            # Find roots
            root1 = optimize.newton(f_y ,guess)
            root2 = optimize.newton(f_y ,-guess)
            
            # Append new roots to array
            locus_points_x.append((root1,skyr_slice*(20/array_size) - 10))
            locus_points_x.append((root2,skyr_slice*(20/array_size) - 10))
        
        # Newton-Raphson Method for x-slices
        if min(m_z[:,skyr_slice]) < 0:
            
            # Find index of closest value to zero and assign NR guess
            guess_indx = np.where(min(abs(m_z[:,skyr_slice])) == abs(m_z[:,skyr_slice]))
            guess = guess_indx[0][0] * (20/array_size) - 10
            
            # Find roots
            root1 = optimize.newton(f_x ,guess)
            root2 = optimize.newton(f_x ,-guess)
            
            # Append new roots to array
            locus_points_y.append((skyr_slice*(20/array_size) - 10, root1))
            locus_points_y.append((skyr_slice*(20/array_size) - 10, root2))
    
    
    # Unpack tuples into array of coordinates
    locus_points_array_x = np.zeros((len(locus_points_x),2))
    locus_points_array_y = np.zeros((len(locus_points_y),2))
    
    indx = 0
    for tuple in locus_points_x:
        locus_points_array_x[indx,0] = tuple[0]
        locus_points_array_x[indx,1] = tuple[1]
        indx += 1
    
    indx = 0
    for tuple in locus_points_y:
        locus_points_array_y[indx,0] = tuple[0]
        locus_points_array_y[indx,1] = tuple[1]
        indx += 1
    
    # Sort locus points array into order by phi    
    locus_points_array_x = locus_points_array_x[np.argsort(np.arctan2(locus_points_array_x[:,0], locus_points_array_x[:,1]))[:]]
    locus_points_array_y = locus_points_array_y[np.argsort(np.arctan2(locus_points_array_y[:,0], locus_points_array_y[:,1]))[:]]
        
    # Use savgol filter on arrays    
    locus_points_array_x[:,0] = savgol_filter(locus_points_array_x[:,0], 5, 3)
    locus_points_array_y[:,1] = savgol_filter(locus_points_array_y[:,1], 5, 3) 
    
    # Join both arrays into one array containing all co-ordinates on boundary
    locus_points_array = np.vstack((locus_points_array_x, locus_points_array_y))
                                         
    # Order into ascending phi order
    locus_points_array = locus_points_array[np.argsort(np.arctan2(locus_points_array[:,0], locus_points_array[:,1]))[:]]                                     
    
    # Repeat first set of co-ordinates at end of array so plot joins up
    locus_points_array = np.vstack((locus_points_array, locus_points_array[0,:]))
    
    
    return locus_points_array

# =============================================================================
# Thiele equation functions:
# =============================================================================


def top_charge(M):
    '''
    This Function obtains the topological charge.
    
    Parameters
    ----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]

    Returns
    -------
    Q : Scalar quantity that is the topological charge of the skyrmion

    '''
    Q = 0
    N = np.int(np.sqrt(len(M)/3))
    for i in range(N):
        for j in range(N):
            sum_ij = 3*(N*i + j)
            sum_up = 3*(N*((i+1)%N) + j)
            sum_right = 3*(N*i + (j+1)%N)
            m_ij = np.array([ M[sum_ij] , M[sum_ij +1] , M[sum_ij +2] ])
            m_right = np.array([ M[sum_right] , M[sum_right +1] , M[sum_right +2] ])
            m_up = np.array([ M[sum_up] , M[sum_up +1] , M[sum_up +2] ])
            
            num = np.dot(m_ij , np.cross(m_right, m_up))
            denom = 1 + np.dot(m_ij , m_right) + np.dot(m_ij , m_up) + np.dot(m_right , m_up)
            q_ijk = 2*np.arctan2(num,denom)
            
            Q+= q_ijk
    
    Q = -2*Q/(4*np.pi)
    return Q

def diss_matrix(M , system_params):
    '''
    This function returns the dissipative matrix
    
    Parameters:
    -----------
    M: Single column vector storing the x,y,z components of all the spins in form:
        [mx_00 , my_00 , mz_00 , ... , mx_NN , my_NN , mz_NN]

    Returns:
    --------
    D : The dissipative matrix
    
    '''
    delta_x = system_params.delta_x
    
    # Calculating the lattice width (in unit cells) from the length of M-vector:
    N = np.int(np.sqrt(len(M)/3))
    # Extracting the m_x, m_y and m_z from the M-vector.
    m_x , m_y , m_z = M_exp(M)[0] , M_exp(M)[1] , M_exp(M)[2]
    
    delta_x = 1
    D = np.zeros(shape = (2,2))
    
    # Building the dissipative matrix
    for i in range(N):
        for j in range(N):

            # Calculating dM/dX components
            dmx_dy = (m_x[(i+1)%N,j] - m_x[(i-1)%N,j]) / (2*delta_x)
            dmy_dy = (m_y[(i+1)%N,j] - m_y[(i-1)%N,j]) / (2*delta_x)
            dmz_dy = (m_z[(i+1)%N,j] - m_z[(i-1)%N,j]) / (2*delta_x)
            
            # Calculating dM/dY components
            dmx_dx = (m_x[i,(j+1)%N] - m_x[i,(j-1)%N]) / (2*delta_x)
            dmy_dx = (m_y[i,(j+1)%N] - m_y[i,(j-1)%N]) / (2*delta_x)
            dmz_dx = (m_z[i,(j+1)%N] - m_z[i,(j-1)%N]) / (2*delta_x)
            
            # Vectors for the partial derivatives
            dm_dx = [dmx_dx, dmy_dx, dmz_dx]
            dm_dy = [dmx_dy, dmy_dy, dmz_dy]
            
            # Update sums for each matrix element
            D[0,0] += np.dot(dm_dx, dm_dx)
            D[0,1] += np.dot(dm_dx, dm_dy)
            D[1,0] += np.dot(dm_dy, dm_dx)
            D[1,1] += np.dot(dm_dy, dm_dy)
    
    return D    

    
def skyr_velocity(Q , diss_mat , system_params):
    '''
    This function calculates the skyrmion velocity vector.

    Parameters
    ----------
    Q : Topological charge
    diss_mat : Dissipative matrix
    alpha : Gilbert damping
    beta : Non-adiabaticity
    j_c : current density vector

    Returns
    -------
    Skyrmion velocity vector

    '''
    alpha = system_params.alpha
    beta = system_params.beta
    j_c = system_params.j_c
    
    detD = np.linalg.det(diss_mat)
    G = 4*np.pi*Q
    prefactor = 1/(G**2 + (alpha**2)*detD)
    Dxx = diss_mat[0 , 0]
    Dxy = diss_mat[0 , 1]
    Dyy = diss_mat[1 , 1]
    inv = np.zeros(shape = (2 , 2))
    inv[0 , 0] = G**2 -(G*Dxy*(beta + alpha)) + (alpha*beta)*(Dxx*Dyy + Dxy**2)
    inv[1 , 1] = G**2 -(G*Dxy*(beta + alpha)) + (alpha*beta)*(Dxx*Dyy + Dxy**2)
    inv[0 , 1] = (G*Dyy*(alpha - beta)) + (2*alpha*beta*Dyy*Dxy)
    inv[1 , 0] = (G*Dxx*(beta - alpha)) + (2*alpha*beta*Dxx*Dxy)
    inv *= prefactor
    return inv.dot(j_c)

def V_STT(system_params, v_sky):
    '''
    Generates a matrix that performs the STT operation (v_sky.∇) on the magnetisation
    texture, where 'v_sky' is the skyrmion velocity vector.

    Parameters
    ----------
    N : Lattice width (in unit cells).
   v_sky: skyrmion velocity vector as determined by Thiele equation.

    Returns
    -------
    V_matrix : Sparse matrix performs (v_sky.∇) operation on every magnetisation vector in the 
    lattice.

    '''
    # Get parameters from system_params object
    N = system_params.array_size
    delta_x = system_params.delta_x
    
    v_x = v_sky[0]
    v_y = v_sky[1]
    V = np.zeros(shape = (3*N**2 , 3*N**2))
    for i in range(N):
        for j in range(N):
            ij = 3*(N*i + j)
            down = 3*(N*((i-1)%N) + j) #<-modulo for periodic boundary conditions.
            up = 3*(N*((i+1)%N) + j)
            right = 3*(N*i + (j+1)%N)
            left = 3*(N*i + (j-1)%N)
            # X-components of magnetisation:
            V[ij , right] = v_x*1*0.5
            V[ij , up] = v_y*1*0.5
            V[ij ,left] = -1*v_x*0.5
            V[ij , down] = -1*v_y*0.5
            # Y-components of magnetisation:
            V[ij+1 , right+1] = 1*v_x*0.5
            V[ij+1 , up+1] = 1*v_y*0.5
            V[ij+1 ,left+1] = -1*v_x*0.5
            V[ij+1 , down+1] = -1*v_y*0.5
            # Z-components of magnetisation:
            V[ij+2 , right+2] = 1*v_x*0.5
            V[ij+2 , up+2] = 1*v_y*0.5
            V[ij+2 ,left+2] = -1*v_x*0.5
            V[ij+2 , down+2] = -1*v_y*0.5
    V/=delta_x
    V_matrix = sparse.csr_matrix(V)
    return V_matrix

def dist_param(diss_matrix):
    '''
    This function takes the dissipative matrix and calculates the distortion
    parameter from the eigenvalues of the dissipative matrix

    Parameters
    ----------
    diss_matrix : The 2x2 dissipative matrix

    Returns
    -------
    distortion : Scalar distortion parameter

    '''
    eigen_values, eigen_vecs = linalg.eig(diss_matrix)
    eigen_values = eigen_values.real
    
    dist = (max(eigen_values) - min(eigen_values)) / max(eigen_values)   
    
    return dist
# @njit
def sus(M):
    N = np.int(np.sqrt(len(M)/3))
    M_sum_x = sum(M[::3])
    M_sum_y = sum(M[1::3])
    M_sum_z = sum(M[2::3])
    chi = np.array([M_sum_x , M_sum_y , M_sum_z])/N
    return chi
    
def skyr_radius(M, system_params):
    '''
    Parameters
    ----------
    m_z : TYPE
        DESCRIPTION.
    system_params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    array_size = system_params.array_size
    L = system_params.L
    
    x_coord = np.linspace(-L/2,L/2,array_size)
    y_coord = np.linspace(-L/2,L/2,array_size)
    
    # Extracting m_z values:
    m_z = M_exp(M)[2]

    # Curve fitting to find function of slice
    poly_order = 15
    fitted_params_y = np.polyfit(x_coord, m_z[int(L/2),:], poly_order)
    fitted_params_x = np.polyfit(y_coord, m_z[:,int(L/2)], poly_order)
        
    # Create function from fitted_parameters
    f_y = np.poly1d(fitted_params_y)
    f_x = np.poly1d(fitted_params_x)    
    
    # Find index of closest value to zero and assign NR guess - YDIRECTION
    guess_indx = np.where(min(abs(m_z[int(L/2),:])) == abs(m_z[int(L/2),:]))
    guess = guess_indx[0][0] * (20/array_size) - 10
            
    # Find roots - YDIRECTION
    root_1 = optimize.newton(f_y ,guess)
    root_2 = optimize.newton(f_y ,-guess)            

    # Now do the same for the x-direction
    # Find index of closest value to zero and assign NR guess - XDIRECTION
    guess_indx = np.where(min(abs(m_z[:,int(L/2)])) == abs(m_z[:,int(L/2)]))
    guess = guess_indx[0][0] * (20/array_size) - 10
            
    # Find roots - XDIRECTION
    root_3 = optimize.newton(f_x ,guess)
    root_4 = optimize.newton(f_x ,-guess)
    
    # Average radius
    radius = (abs(root_1) + abs(root_2) + abs(root_3) + abs(root_4)) / 4
    
    return radius