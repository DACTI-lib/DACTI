input0 = inputs[0]

import numpy as np

rho = input0.CellData["p_0"]
u = input0.CellData["p_1"]
v = input0.CellData["p_2"]
w = input0.CellData["p_3"]
p = input0.CellData["p_4"]

T = p / (rho * 287.0)
a = np.sqrt(1.4 * p / rho)
mach = np.sqrt(u**2 + v**2 + w**2) / a

velocity = np.zeros((len(u), 3))
velocity[:,0] = u
velocity[:,1] = v
velocity[:,2] = w

output.CellData.append(velocity, "Velocity")
output.CellData.append(mach, "Mach")
output.CellData.append(p, "Pressure")
output.CellData.append(rho, "Density")

output.CellData.append(input0.CellData["acti_level"], "ACTI Level")

