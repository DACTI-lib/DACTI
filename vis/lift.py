import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

gamma = 1.4

def beta_theta_mach_relation(beta, theta, M1):
    beta = np.deg2rad(beta)
    theta = np.deg2rad(theta)
    lhs = np.tan(theta)
    rhs = 2 / np.tan(beta) * ((M1**2 * np.sin(beta)**2 - 1) / 
                              (M1**2 * (gamma + np.cos(2 * beta)) + 2))
    return lhs - rhs

def prandtl_meyer_function(M):
    sqrt_term = np.sqrt((gamma + 1) / (gamma - 1))
    term1 = sqrt_term * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term2 = np.arctan(np.sqrt(M**2 - 1))
    return np.rad2deg(term1 - term2)

def find_shock_angle(theta, M1):
    # Initial guess for beta in degrees
    beta_initial_guess = np.rad2deg(np.arcsin(1/M1)) + theta
    
    # Solve for beta
    beta_solution = fsolve(beta_theta_mach_relation, beta_initial_guess, args=(theta, M1))
    
    return beta_solution[0]

def find_mach_number(nu, M_initial_guess):
    def pm_eq(M):
        return prandtl_meyer_function(M) - nu
    
    # Solve for Mach number
    M_solution = fsolve(pm_eq, M_initial_guess)
    
    return M_solution[0]

def post_shock_state(beta, M1):
    beta = np.deg2rad(beta)
    p = 1 + (2 * gamma) / (gamma + 1) * (M1**2 * np.sin(beta)**2 - 1)
    r = (gamma + 1) * M1**2 * np.sin(beta)**2 / (2 + (gamma - 1) * M1**2 * np.sin(beta)**2)

    return p

def post_shock_mach_number(theta, beta, M1):
    beta = np.deg2rad(beta)
    theta = np.deg2rad(theta)
    rhs = (1 + (gamma - 1)/(gamma + 1) * (M1**2 * np.sin(beta)**2 - 1)) / (1 + 2 * gamma/(gamma + 1) * (M1**2 * np.sin(beta)**2 - 1)) 
    M2 = np.sqrt(rhs / np.sin(beta - theta)**2)

    return M2

def post_expansion_state(M1, M2):
    P1 = 1 + 0.5 * (gamma - 1) * M1**2
    P2 = 1 + 0.5 * (gamma - 1) * M2**2
    return (P1 / P2) ** (gamma / (gamma - 1))

def infinitely_thin_plate(alpha, mach):
    M1 = mach
    theta = alpha
    
    # Expansion fan on top surface
    nu = prandtl_meyer_function(M1) + theta
    M2 = find_mach_number(nu, M1)
    p2 = post_expansion_state(M1, M2)

    # Oblique shock on bottom surface
    beta = find_shock_angle(theta, M1)
    p3 = post_shock_state(beta, M1)

    Cl = (p3 - p2) * np.cos(np.deg2rad(theta)) / (0.5 * gamma * M1**2)
    Cd = Cl * np.tan(np.deg2rad(theta))

    return Cl

def diamond(alpha, mach, width, height):
    M1 = mach
    e = np.rad2deg(np.arctan(height / width))

    M2 = 0.0
    p2 = 0.0

    if (alpha - e) < 0:
        # First shock wave on top surface
        theta_2 = e - alpha
        beta = find_shock_angle(theta_2, M1)
        p2 = post_shock_state(beta, M1)
        M2 = post_shock_mach_number(theta_2, beta, M1)
    else:
        # First Expansion fan on top surface
        theta_2 = alpha - e
        nu2 = prandtl_meyer_function(M1) + theta_2
        M2 = find_mach_number(nu2, M1)
        p2 = post_expansion_state(M1, M2)

    # Second Expansion fan on top surface
    theta_3 = 2 * e
    nu3 = prandtl_meyer_function(M2) + theta_3
    M3 = find_mach_number(nu3, M2)
    p3 = post_expansion_state(M2, M3) * p2

    # first Oblique shock on bottom surface
    theta_4 = alpha + e
    beta = find_shock_angle(theta_4, M1)
    p4 = post_shock_state(beta, M1)
    M4 = post_shock_mach_number(theta_4, beta, M1)

    # First Expansion fan on bottom surface
    theta_5 = 2 * e
    nu5 = prandtl_meyer_function(M4) + theta_5
    M5 = find_mach_number(nu5, M4)
    p5 = post_expansion_state(M4, M5) * p4

    # normal and tangential forces 
    T = 0.5 * height * (p2 + p4 - p3 - p5)
    N = 0.5 * width  * (p4 + p5 - p3 - p2)

    # lift and drag
    a = np.deg2rad(alpha)
    L = N * np.cos(a) - T * np.sin(a)
    D = N * np.sin(a) + T * np.cos(a)

    Cl = L / (0.5 * gamma * M1**2 * width) 
    Cd = D / (0.5 * gamma * M1**2 * width)

    return Cl