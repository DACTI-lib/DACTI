import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
bwidth = 2.0
font_size  = 15
label_size = 15
marker_size= 4
line_width = 1.2
legend_size= 13

#--------------------------------------------------------------------------#
# data
#--------------------------------------------------------------------------#

saveplot  = 0

data = pd.read_csv('/Users/yjun/Codes/DACTI/cases/out/oscDrop_observables.csv')

t = data['time']
ke = data['ke']
Etot = data['Etot']

#--------------------------------------------------------------------------#
# plot
#--------------------------------------------------------------------------#
fig, ax = plt.subplots()
ax.plot(t, ke, 'ko', markersize = marker_size)
# ax.plot(t, Etot, 'ko', markersize = marker_size)
# ax.set_yscale('log')
ax.spines['top'].set_linewidth(bwidth)
ax.spines['bottom'].set_linewidth(bwidth)
ax.spines['left'].set_linewidth(bwidth)
ax.spines['right'].set_linewidth(bwidth)
ax.axes.xaxis.set_ticks_position('both')
ax.tick_params(axis='x', direction='in', labelsize=label_size)
ax.axes.yaxis.set_ticks_position('both')
ax.tick_params(axis='y', direction='in', labelsize=label_size)

plt.show()