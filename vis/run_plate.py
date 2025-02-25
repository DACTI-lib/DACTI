
import subprocess
from pathlib import Path
import argparse
import tomllib
import pandas as pd
from plotlab import plotlab
import lift

import matplotlib.pyplot as plt
import numpy as np

font_size = 15

DACTI_EXEC_PATH = Path("build/aero_test_2D")
DACTI_CASE_FILE_PATH = Path("cases")
DACTI_OUT_PATH = Path("cases/out")
DACTI_PLOT_OUT_PATH = Path("vis/out")

parser = argparse.ArgumentParser(description="Run aero case")
parser.add_argument("--postprocess", type=bool, default=False, help="only plot")

args = parser.parse_args()

case_file_name = "plate.tomlt"

def run_case(**kwargs):
    case_file_path = DACTI_CASE_FILE_PATH / case_file_name

    with open(case_file_path, "r") as f:
        case_file_content = f.read()
        case_file_content = case_file_content.format(**kwargs)

        case_file_instance = tomllib.loads(case_file_content)

    case_instance_name = case_file_instance["scene"]["name"]

    if not args.postprocess:
        case_file_instance_path = DACTI_CASE_FILE_PATH / (case_file_instance["scene"]["name"] + ".toml")

        with open(case_file_instance_path, "w") as f:
            f.write(case_file_content)
        
        subprocess.run([str(DACTI_EXEC_PATH), str(case_file_instance_path)], check=True)

    return pd.read_csv(DACTI_OUT_PATH / (case_instance_name + "_observables.csv"))

alpha = 13
machs = [2.8, 3.0, 3.2]
levels = [6,7,8,9,10]

###
# analytical solution
###
c_l_analytic = []

for mach in machs:
    c_l_ = lift.infinitely_thin_plate(alpha, mach)
    c_l_analytic.append(c_l_)

machs_ = np.linspace(2.8, 3.2, 10)
c_l_analytic_ = np.array(
    [lift.infinitely_thin_plate(alpha, machs_[i]) for i in range(10)]
)

plab = plotlab()
fig1, ax1 = plt.subplots()
plab.format_ax(ax1)
ax1.plot(machs_, c_l_analytic_, label='analytical', color = 'k')

# colors = [
#     # ( 33/255, 49/255,140/255),
#     ( 30/255,128/255,184/255),
#     (0.1924, 0.6393, 0.3295),
#     (0.9866, 0.7471, 0.1302)
# ]

colors = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#ff7f0e'
]

error = np.empty((len(levels), len(machs)))

for i in range(len(levels)):
    c_l = []

    for mach in machs:
        time_series = run_case(mach=mach, level=levels[i])

        c_l_series = time_series["Cl"].to_numpy()
        c_d_series = time_series["Cd"].to_numpy()

        c_l.append(c_l_series[-1])

    if i == len(levels) - 1:
        ax1.plot(machs, c_l, 'ko', 
                 label='numerical', 
                 markerfacecolor='white')

    error[i,:] = np.abs(np.array(c_l_analytic) - np.array(c_l))

plt.tick_params(axis="both", labelsize=font_size)
ax1.set_xlabel("$Ma_1$", fontsize=font_size)
ax1.set_xticks(machs)
ax1.set_ylabel("$C_L$", fontsize=font_size)
ax1.legend(loc='best', frameon=False, fontsize=font_size)

figname = f"{DACTI_PLOT_OUT_PATH / case_file_name}.pdf"

plt.savefig(figname)
print(f"Result figure saved to {figname}")


# convergence plot
fig2, ax2 = plt.subplots()
plab.format_ax(ax2)

for i in range(len(machs)):
    ax2.semilogy(levels, error[:,i], '--o', 
                 label=f'$Ma$ = {machs[i]}',
                 color=colors[i],
                 markerfacecolor='white')

# 2nd order
error_t = [error[0,0] / (4**i) for i in range(len(levels))] 

ax2.semilogy(levels, error_t, 'k-', label="$\mathcal{O}(R_{\max}^{-2})$")
plt.tick_params(axis="both", labelsize=font_size)
plt.minorticks_off()
ax2.set_xticks(levels)
ax2.set_xlabel("$R_{\max}$", fontsize=font_size)
ax2.set_ylabel("$L_1$-error", fontsize=font_size)
ax2.legend(loc='best', frameon=False, fontsize=font_size)

figname = f"{DACTI_PLOT_OUT_PATH / case_file_name}_convergence.pdf"

plt.savefig(figname)
print(f"Result figure saved to {figname}")