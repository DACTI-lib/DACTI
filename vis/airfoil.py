import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Airfoil:

    def __init__(self, c, m, t, δ, Uinf, α):
        """
        Initialize transformation variables
        """
        xc = -1/1.299 * t
        yc =  2       * m
        β  = np.arcsin(yc)
        a  = np.cos(β) + xc
        z0 = xc + 1j*yc
        n  = 2 - δ/np.pi

        LE = np.e**(1j*( β + np.pi)) + z0
        TE = np.e**(1j*(-β        )) + z0
        ζ1 = n*a*((LE+a)**n + (LE-a)**n)/((LE+a)**n - (LE-a)**n)
        ζ2 = n*a*((TE+a)**n + (TE-a)**n)/((TE+a)**n - (TE-a)**n)

        self.c     = c
        self.β     = β
        self.a     = a
        self.z0    = z0
        self.n     = n
        self.shift = 0.5 * (ζ1 + ζ2)                 # x-shift to make LE the origin
        self.c_map = ζ2 - ζ1            # chord of transformed Karman-Trefftz airfoil
        self.Uinf  = Uinf               # free stream velocity
        self.α     = α /180*np.pi       # angle of attack

        print("a = ", a)
        print("n = ", n)
        print("beta = ", β)
        print("LE = ", ζ1)
        print("TE = ", ζ2)
        print("chord = ", self.c_map)

    def Map(self, z, Jacob=False):
        """
        Mapping that transforms a unit circle in z-space to an airfoil in ζ-space. \\
        The mapped airfoil is scaled to the prescribed chord length and shifted such LE coincides with origin.
        """
        a  = self.a
        n  = self.n
        ζ  = n*a*((z +a)**n + (z -a)**n)/((z +a)**n - (z -a)**n)    # Karman-Trefftz tranform                                       
        ζ  = self.c/self.c_map * (ζ - self.shift)                   # shift and normalize
        J  = self.c/self.c_map / (4*(a*n)**2*(z**2-a**2)**(n-1)/((z+a)**n-(z-a)**n)**2)    # Jacobian of the transform
        
        if Jacob: return J
        else:     return ζ

    def inverse_Map(self, ζ):
        ## only works for Joukowski transform with n = 2
        z = np.zeros_like(ζ)

        ζ = self.c_map/self.c * ζ + self.shift

        for i in range(ζ.shape[0]):
            for j in range(ζ.shape[1]):
                if (ζ[i, j].real > 0):
                    z[i, j] = 0.5 * (ζ[i, j] + np.sqrt(ζ[i, j]**2 - 4.0 * self.a**2))
                else:
                    z[i, j] = 0.5 * (ζ[i, j] - np.sqrt(ζ[i, j]**2 - 4.0 * self.a**2))
        
        return z

    def Gen_airfoil(self, scope:str='whole', N:int=100):
        """
        Returns the coordinates of the airfoil in ζ-space as a complex number. \\
        `N`: number of control points, default value = 100; \\
        `scope = whole`: returns the entire airfoil, default option; \\
        `scope = upper`: returns only the upper surface.
        """
        if scope == 'whole': θ = np.linspace(0    , 2*np.pi, N)
        else:                θ = np.linspace(np.pi, 0      , N) # from leading to trailing edge

        z = np.e**(1j*θ) + self.z0
        ζ = self.Map(z)
        return ζ, z

    def complex_potential(self, z):
        Uinf = self.Uinf * self.c_map/self.c    # scale free stream velocity
        α    = self.α
        β    = self.β
        z0   = self.z0
        Γ    = 4*np.pi * Uinf * np.sin(α+β)     # circulation - Kutta condition
    
        return Uinf*(z-z0)*np.e**(-1j*α) + Uinf/(z-z0)*np.e**(1j*α) + 1j*Γ/np.pi/2*np.log(z-z0)

    def complex_velocity(self, z):
        Uinf = self.Uinf * self.c_map/self.c    # scale free stream velocity
        α    = self.α
        β    = self.β
        z0   = self.z0
        Γ    = 4*np.pi * Uinf * np.sin(α+β)     # circulation - Kutta condition
        J    = self.Map(z, Jacob=True)

        return (Uinf*np.e**(-1j*α) - Uinf/(z-z0)**2*np.e**(1j*α) + 1j*Γ/(z-z0)/np.pi/2) * J

    def calc_flow_field(self):
        """
        Calculates the 2D flow field using potential flow theory.
        """
        # Generate 2D Cartesian mesh in z-space
        # Ub, Bb = self.z0.imag + 5, self.z0.imag - 4     # upper and bottom bound
        # Rb, Lb = self.z0.real + 5, self.z0.real - 4     # right and left bound
        # xx, yy = np.meshgrid(np.linspace(Lb, Rb, 1000), 
        #                      np.linspace(Bb, Ub, 1000) )
        # zz = xx + yy*1j                 
        # ζζ = self.Map(zz)
        
        xx, yy = np.meshgrid(np.linspace(-1, 1, 1000), 
                             np.linspace(-1, 1, 1000) )
        
        ζζ = xx + yy*1j
        zz = self.inverse_Map(ζζ)

        F  = self.complex_potential(zz)
        W  = self.complex_velocity(zz)

        # remove values inside the circle -> necessary, otherwise get unphysical values
        F[abs(zz-self.z0) < 1] = "nan"
        W[abs(zz-self.z0) < 1] = "nan"

        return zz, ζζ, F, W
    
    def Ufree(self, X):
        """
        Return free stream velocity
        """
        ns   = 3*np.size(X)
        ζ, z = self.Gen_airfoil(scope='upper', N=ns)
        ds   = np.sqrt( (np.diff(ζ.real))**2 + (np.diff(ζ.imag))**2 )
        s    = np.zeros(ns)
        for i in range(1, ns): s[i] = s[i-1] + ds[i-1]

        Uabs = abs(self.complex_velocity(z))    # velocity magnitude
        Us   = interp1d(s, Uabs, kind='nearest')

        return Us(X)
   
    def dpdx(self, X):
        """
        Return pressure gradient along main stream direction. \\
        `X`: wall coordinates along airfoil surface. Should be within the range of upper surface
        """
        # define a wall coordinate system (s,n) along upper airfoil surface
        ns   = 3*np.size(X)
        ζ, z = self.Gen_airfoil(scope='upper', N=ns)
        ds   = np.sqrt( (np.diff(ζ.real))**2 + (np.diff(ζ.imag))**2 )
        s    = np.zeros(ns)
        for i in range(1, ns): s[i] = s[i-1] + ds[i-1]

        # calculate pressure gradient
        Uabs = abs(self.complex_velocity(z))    # velocity magnitude
        dpds = -0.5*(Uabs[1:]**2 - Uabs[0:-1]**2) / np.diff(s)
        dPs  = interp1d(s, dpds, kind='nearest')

        return dPs(X)

# @click.command()
# @click.option(
#     "--c", type=float, default=1.0, help="Number of gridpoints in x-direction"
# )
# @click.option(
#     "--plot_result", type=bool, default=True, help="Make a plot of the result?"
# )
def main(plot_result=True):
    c = 1
    m    = 0.0
    t    = 0.12
    δ    = 0
    Ma   = 0.01
    Uinf = 0.1 #Ma * np.sqrt(1.4)
    α    = 5
    AF   = Airfoil(c, m, t, δ/180*np.pi, Uinf, α)
    surf, zsurf = AF.Gen_airfoil('whole')

    zz, ζζ, F, W = AF.calc_flow_field()

    if plot_result:
        fig, ax = plt.subplots()
        cbar = ax.pcolormesh(ζζ.real, ζζ.imag, W.real)
        # ax.set_xlim(-0.5, 1.5)
        # ax.set_ylim(-0.5, 0.5)
        ax.set_aspect(1)
        plt.colorbar(cbar)
        plt.show()


if __name__ == "__main__":
    main()