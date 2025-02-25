input0 = inputs[0]

import numpy as np

rho = input0.PointData["u_0"]
rho_u = input0.PointData["u_1"]
rho_v = input0.PointData["u_2"]
rho_E = input0.PointData["u_3"]

u = rho_u / rho
v = rho_v / rho

gamma = 1.4

p = (gamma - 1.0) * (rho_E - 0.5 * rho * (u**2 + v**2))

T = p / (rho * 287.0)

a = np.sqrt(gamma * p / rho)

mach = np.sqrt(u**2 + v**2) / a

velocity_array = np.zeros((len(u), 2))

velocity_array[:,0] = u
velocity_array[:,1] = v

output.PointData.append(velocity_array, "Velocity")
output.PointData.append(mach, "Mach")
output.PointData.append(T, "Temperature")
output.PointData.append(p, "Pressure")
output.PointData.append(rho, "Density")
output.PointData.append(a, "Speed of Sound")
output.PointData.append(rho_E, "Total Energy")

