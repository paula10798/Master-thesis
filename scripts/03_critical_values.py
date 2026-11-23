import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from thesis.estimators.critical_values import (
    simulate_BM,
    Eta_1_Eta_2_calc,
    make_objective_constraint,
    make_objective_constraint_basic,
    hdi,
    compute_eta_inverse,
    compute_eta_inverse_basic,
)

## Critical values for OK extensions ##

# --- Run the analysis as before ---
simulation_100 = simulate_BM(100, 100, seed=123)
results_eta_1_2 = Eta_1_Eta_2_calc(simulation_100)

rs_pairs = [(0.5, 0.5), (2, 1), (3, 1), (1, 3)]
cred_masses = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
labels = [f"({r},{s})" for r,s in rs_pairs]
eta_inverses = []
hdi_95s = []
opt_weights_rs = []

for r, s in rs_pairs:
    objective, constraint = make_objective_constraint(results_eta_1_2, r, s)
    result = minimize(objective, [1.0, 1.0], constraints=constraint)
    l1, l2 = result.x
    opt_weights_rs.append((r, s, l1, l2))
    eta_inv = compute_eta_inverse(results_eta_1_2, l1, l2, r, s)
    eta_inverses.append(eta_inv)
    hdi_95s.append(hdi(eta_inv, 0.95))

    print(f"Optimal lambda_1 for r={r}, s={s}: {l1:.6f}")
    print(f"Optimal lambda_2 for r={r}, s={s}: {l2:.6f}")
    for cm in cred_masses:
        hdi_min, hdi_max = hdi(eta_inv, cm)
        print(f"{int(cm*100)}% HDI: ({hdi_min:.6f}, {hdi_max:.6f})")
    print()


# Simple OK
objective, constraint = make_objective_constraint_basic(results_eta_1_2)
result = minimize(objective, [1.0, 1.0], constraints=constraint)
l1, l2 = result.x
opt_weights_rs.append((None, None, l1, l2))
eta_inv_basic = compute_eta_inverse_basic(results_eta_1_2, l1, l2)
eta_inverses.append(eta_inv_basic)
hdi_95s.append(hdi(eta_inv_basic, 0.95))
labels.append("Simple OK")
print("Simple OK estimator (no lag, power=1):")
print(f"Optimal lambda_1: {l1:.6f}")
print(f"Optimal lambda_2: {l2:.6f}")
for cm in cred_masses:
    hdi_min, hdi_max = hdi(eta_inv, cm)
    print(f"{int(cm*100)}% HDI: ({hdi_min:.6f}, {hdi_max:.6f})")

hdi_95_basic = hdi_95s[-1]

opt_weights_df = pd.DataFrame(opt_weights_rs, columns=["r", "s", "lambda_1", "lambda_2"])
print(opt_weights_df)

# --- Plotting ---

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for idx, label in enumerate(labels[:-1]):  # Only the first 4, each is (r,s)
    ax = axs[idx // 2, idx % 2]

    # PDF and HDI for simple OK
    kde_basic = gaussian_kde(eta_inv_basic)
    x_basic = np.linspace(min(eta_inv_basic), max(eta_inv_basic), 500)
    ax.plot(x_basic, kde_basic(x_basic), label='Simple OK', color='blue')
    ax.axvline(hdi_95_basic[0], color='blue', linestyle='--', label='Simple OK 95% HDI')
    ax.axvline(hdi_95_basic[1], color='blue', linestyle='--')

    # PDF and HDI for current (r,s)
    eta_inv_rs = eta_inverses[idx]
    hdi_95_rs = hdi_95s[idx]
    kde_rs = gaussian_kde(eta_inv_rs)
    x_rs = np.linspace(min(eta_inv_rs), max(eta_inv_rs), 500)
    ax.plot(x_rs, kde_rs(x_rs), label=f'(r,s)={label}', color='orange')
    ax.axvline(hdi_95_rs[0], color='orange', linestyle='--', label=f'(r,s)={label} 95% HDI')
    ax.axvline(hdi_95_rs[1], color='orange', linestyle='--')

    ax.set_xlabel(r'$\xi(\lambda)^{-1}$')
    ax.set_ylabel('Density')
    ax.legend()

# fig.suptitle("Density and HDI", fontsize=16)
plt.tight_layout()
fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.15)
plt.show()
