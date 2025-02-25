input0 = inputs[0]

import numpy as np

rho = input0.PointData["p_0"]
u = input0.PointData["p_1"]
v = input0.PointData["p_2"]
p = input0.PointData["p_3"]

T = p / (rho * 287.0)
a = np.sqrt(1.4 * p / rho)
mach = np.sqrt(u**2 + v**2) / a

velocity = np.zeros((len(u), 3))
velocity[:,0] = u
velocity[:,1] = v

entropy = np.log(1.4/0.4*p/rho ) - 0.4/1.4*np.log(p);

output.PointData.append(velocity, "Velocity")
output.PointData.append(mach, "Mach")
output.PointData.append(T, "Temperature")
output.PointData.append(p, "Pressure")
output.PointData.append(rho, "Density")
output.PointData.append(a, "Speed of Sound")
output.PointData.append(entropy, "Entropy")

output.CellData.append(input0.CellData["acti_level"], "ACTI Level")

