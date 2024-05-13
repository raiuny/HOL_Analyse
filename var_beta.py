from HOL_model import HOL_Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
cmap = mpl.cm.get_cmap('viridis')

def plot_fig(x, y, label, opt_mark = "x" , color = "green", num = 0):
    idx = np.argmin(y)
    pltx, plty = [], []
    for xi, yi in zip(x, y):
        if yi < 2000:
            pltx.append(xi)
            plty.append(yi)
    plt.plot(x[idx], y[idx], marker = opt_mark, color = color)
    plt.scatter(pltx, plty, marker=".", color = color)
    plt.plot(pltx, plty, label=label, linestyle="--", color = color)
    # plt.scatter(x, y)
if __name__ == "__main__":
    tau_T = 32
    tau_F = 27
    nmld = 20
    result = {}
    states = {}
    arrival_rate_mld = 0.002
    arrival_rate_sld = 0.0001
    beta_range = np.arange(0.01, 1.0, 0.01)
    nsld1_range = [0, 4, 10]
    for nsld1 in nsld1_range:
        nsld2 = 20 - nsld1
        result[nsld1]=[]
        states[nsld1]=[]
        for beta in beta_range:
            # link1
            model1 = HOL_Model(
                n1 = nmld,
                n2 = nsld1,
                lambda1 = tau_T * arrival_rate_mld * beta,
                lambda2 = tau_T * arrival_rate_sld,
                W_1 = 16,
                W_2 = 16,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            model2 = HOL_Model(
                n1 = nmld,
                n2 = nsld2,
                lambda1 = tau_T * arrival_rate_mld * (1-beta),
                lambda2 = tau_T * arrival_rate_sld,
                W_1 = 16,
                W_2 = 16,
                K_1 = 6,
                K_2 = 6,
                tt = tau_T,
                tf = tau_F
                )
            result[nsld1].append(model1.e2e_delay[0] * beta + model2.e2e_delay[0] * (1-beta))
            states[nsld1].append(model1.state + '|' + model2.state)
    for k, v in result.items():
        print(k, v)
    for k, v in states.items():
        print(k, v)
    
    for i, c in zip(nsld1_range, ["g", "r", "b"]):
        plot_fig(beta_range, result[i], label=f"{i}:{20-i}", color=c)
    plt.ylabel("E2E delay")
    plt.xlabel(r"$\beta$")
    plt.legend()
    plt.show()