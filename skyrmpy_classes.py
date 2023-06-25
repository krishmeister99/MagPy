import numpy as np
from scipy import sparse


class System(object):
    def __init__(self, J, Z,K, D, array_size, alpha, beta, B, j_c, L):
        self.J = J # Exchange constant
        self.Z = Z # Zeeman Coupling
        self.K = K # Uniaxial anisotropy
        self.D = D # Dzyaloshinskii Moriya interaction
        self.array_size = array_size #Number of lattice points in system.
        self.alpha = alpha # Gilbert damping constant
        self.beta = beta # Non-adiabaticity for STT
        self.B = B #External field unit vector
        self.j_c = j_c #Current density
        self.L = L #System length
        self.delta_x = L/array_size
    
    def sk_initial_condition(self,m, g, tol):
        """
        This method generates a Skyrmion profile using the relaxation method to solve the differential equation for the Theta angle. 
        It returns meshgrids m_x , m_y , m_z corresponding to the x, y and z components of the magnetisation.

        Parameters
        ----------
        m: Sign of the central spin.
        g: Vorticity.
        tol: Tolerance for convergence test.

        Returns
        -------
        mx : x-component of magnetisation at each lattice site
        my : y-component of magnetisation at each lattice site
        mz : z-component of magnetisation at each lattice site
        
        """

        Z = self.Z
        J = self.J
        D = self.D
        array_size = self.array_size
        L = self.L
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

            if np.max(np.abs(Theta - Theta_last)) <= tol:
                print("Solution converged in "+ str(k) + " steps")
                not_converge = False
                break
            k += 1
        
        X,Y = np.meshgrid(x_coord,y_coord)
        m_x = np.sin(Theta) * np.cos(m*phi(X,Y)+g)
        m_y = np.sin(Theta) * np.sin(m*phi(X,Y)+g)    
        m_z = np.cos(Theta)

        return m_x , m_y , m_z

    
