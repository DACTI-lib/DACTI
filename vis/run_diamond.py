
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
# parser.add_argument("--case", type=str, help="Case name (e.g., diamond.tomlt)")
parser.add_argument("--postprocess", type=bool, default=False, help="only plot")

args = parser.parse_args()

# case_file_name = Path(args.case)
case_file_name = "diamond.tomlt"

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

# alphas = range(1,17,2)
alphas = [5, 10, 15]
width = 1
height = 0.176
mach = 2.5
levels = [6,7,8,9,10]

colors = [
    '#1f77b4',
    '#d62728',
    '#2ca02c',
    '#ff7f0e'
]

###
# analytical solution
###
c_l_analytic = []

for angle in alphas:
    c_l_= lift.diamond(angle, mach, width, height)
    c_l_analytic.append(c_l_)


plab = plotlab()
fig1, ax1 = plt.subplots()
plt.tick_params(axis="both", labelsize=font_size)
plab.format_ax(ax1)
ax1.plot(alphas, c_l_analytic, label='analytical', color = 'k')


error = np.empty((len(levels), len(alphas)))

for i in range(len(levels)):
    c_l = []

    for angle in alphas:
        time_series = run_case(angle=angle, level=levels[i])

        c_l_series = time_series["Cl"].to_numpy()
        c_d_series = time_series["Cd"].to_numpy()

        c_l.append(c_l_series[-1]*np.cos(np.deg2rad(angle)) - c_d_series[-1]*np.sin(np.deg2rad(angle)))

    if i == len(levels) - 1:
        ax1.plot(alphas, c_l, 'ko', 
                 markerfacecolor='white',
                 label='numerical')

    error[i,:] = np.abs(np.array(c_l_analytic) - np.array(c_l))

ax1.set_xticks(alphas)
ax1.set_xlabel(r"$\alpha~[^\circ]$", fontsize=font_size)
ax1.set_ylabel("$C_L$", fontsize=font_size)
ax1.legend(loc='best', frameon=False, fontsize=font_size)

figname = f"{DACTI_PLOT_OUT_PATH / case_file_name}.pdf"

plt.savefig(figname)
print(f"Result figure saved to {figname}")

fig2, ax2 = plt.subplots()
plab.format_ax(ax2)
plt.tick_params(axis="both", labelsize=font_size)

alpha_str = r'$\alpha$'
deg_ = r'$^{\circ}$'
for i in range(len(alphas)):
    ax2.semilogy(levels, error[:,i], '--o', 
                 color=colors[i],
                 markerfacecolor="white",
                 label=f'{alpha_str} = {alphas[i]}{deg_}')

# 2nd order
error_t = [error[0,0] / (4**i) for i in range(len(levels))] 
ax2.semilogy(levels, error_t, 'k-', label="$\mathcal{O}(R_{\max}^{-2})$")
plt.minorticks_off()
ax2.set_xticks(levels)
ax2.set_xlabel("$R_{\max}$", fontsize=font_size)
ax2.set_ylabel("$L_1$-error", fontsize=font_size)
ax2.legend(loc='best', frameon=False, fontsize=font_size)

figname = f"{DACTI_PLOT_OUT_PATH / case_file_name}_convergence.pdf"

plt.savefig(figname)
print(f"Result figure saved to {figname}")