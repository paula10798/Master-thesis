import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis.estimators.power_variation import proposed_estimator, RV_estimator, RPV_estimator
from thesis.evaluation import bias, RMSE, ci_and_coverage_ratio, true_volatility_calculate

## Monte Carlo Simulation for Weekend and Overnight returns ##

print('We will now be running Monte Carlo Simulations, using the previous models')

################################
# Simulating Overnight Returns #
################################

print("-----------------")
print('For Overnight returns, we simulate the MAR (1) model with Student-t and Gaussian distributions')


np.random.seed(42)

from scipy.stats import t as t_dist

N = 1000        # Number of observations
M = 1000        # Number of simulations
p_G = 0.6       # Mixture weight for Gaussian
phi1 = 0.7      # AR(1) coefficient for Gaussian
phi2 = 0.3      # AR(1) coefficient for Student-t

# Gaussian component parameters
mu_G, sigma_G = 0, 1
params_gaussian1 = [mu_G, sigma_G]

# Student-t component parameters
mu_T, sigma_T, df_T = 0, 1.5, 5
params_t1 = [mu_T, sigma_T, df_T]

mar1_simulations = np.zeros((M, N))
components = np.zeros((M, N), dtype=int)

for m in range(M):
    mar1_simulations[m, 0] = np.random.normal()
    for t in range(1, N):
        # Choose component
        if np.random.rand() < p_G:
            components[m, t] = 0
            noise = np.random.normal(mu_G, sigma_G)
            mar1_simulations[m, t] = phi1 * mar1_simulations[m, t-1] + noise
        else:
            components[m, t] = 1
            noise = t_dist.rvs(df_T, loc=mu_T, scale=sigma_T)
            mar1_simulations[m, t] = phi2 * mar1_simulations[m, t-1] + noise

mar1_simulations_df = pd.DataFrame(mar1_simulations)

print('We now display the simulated MAR (1) model for the first 5 simulations')
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.plot(mar1_simulations[i], label=f'Sim {i+1}')
plt.title('MAR(1) Model Process, First 5 Simulations')
plt.xlabel('Time step')
plt.ylabel('Log-return $(r_t)$')
plt.legend()
plt.show()

print('---------------------')

#------------------------#
# Computing volatility estimates, using simulated data
#------------------------#

# Estimates for Proposed estimator with lag 1 and r = 2

r2_l1 = 2    # power r
w_l1 = [0.4] # Autocorrelation coefficients

estimator1_df_mar = pd.DataFrame({
    col: proposed_estimator(mar1_simulations_df[col], r2_l1, w_l1) for col in mar1_simulations_df.columns
    })

# Estimates for Proposed estimator with lag 1 and r = 0.5

r05_l1 = 0.5
w_l1 = [0.4]

estimator2_df_mar = pd.DataFrame({
    col: proposed_estimator(mar1_simulations_df[col], r05_l1, w_l1) for col in mar1_simulations_df.columns
    })

# Estimates for Proposed estimator with lag 1 and r = 3

r3_l1 = 3
w_l1 = [0.4]

estimator6_df_mar = pd.DataFrame({
    col: proposed_estimator(mar1_simulations_df[col], r3_l1, w_l1) for col in mar1_simulations_df.columns
    })

# Estimates for RV

estimator3_df_mar = pd.DataFrame({
    col: RV_estimator(mar1_simulations_df[col]) for col in mar1_simulations_df.columns
    })

# Estimates for RPV with r = 2

r2_RPV = 2

estimator4_df_mar = pd.DataFrame({
    col: RPV_estimator(mar1_simulations_df[col], r2_RPV) for col in mar1_simulations_df.columns
    })

# Estimates for RPV with r = 0.5

r05_RPV = 0.5

estimator5_df_mar = pd.DataFrame({
    col: RPV_estimator(mar1_simulations_df[col], r05_RPV) for col in mar1_simulations_df.columns
    })

# Estimates for RPV with r = 3

r3_RPV = 3

estimator7_df_mar = pd.DataFrame({
    col: RPV_estimator(mar1_simulations_df[col], r3_RPV) for col in mar1_simulations_df.columns
    })


#-------------------------#
# Evaluating the performance of estimators
#-------------------------#


print('We are now evaluating the performance of the estimators under the simulated MAR (1) Model.')

true_volatility_mar = true_volatility_calculate(params_gaussian1, params_t1, p_G)

print('----------------------')
# For Proposed estimator with lag 1 and r = 2
estimates_pe_l1_r2 = estimator1_df_mar.mean(axis=1).values
bias_pe_l1_r2 = bias(estimates_pe_l1_r2, true_volatility_mar)
rmse_pe_l1_r2 = RMSE(estimates_pe_l1_r2, true_volatility_mar)
coverage_pe_l1_r2, lower_ci_pe_l1_r2, upper_ci_pe_l1_r2 = ci_and_coverage_ratio(estimator1_df_mar, true_volatility_mar)

print("Proposed estimator with lag 1 and r = 2:")
print("Bias:", bias_pe_l1_r2)
print("RMSE:", rmse_pe_l1_r2)
print("Coverage ratio:", coverage_pe_l1_r2)

print('----------------------')
# For Proposed estimator with lag 1 and r = 0.5
estimates_pe_l1_r05 = estimator2_df_mar.mean(axis=1).values
bias_pe_l1_r05 = bias(estimates_pe_l1_r05, true_volatility_mar)
rmse_pe_l1_r05 = RMSE(estimates_pe_l1_r05, true_volatility_mar)
coverage_pe_l1_r05, lower_ci_pe_l1_r05, upper_ci_pe_l1_r05 = ci_and_coverage_ratio(estimator2_df_mar, true_volatility_mar)

print("Proposed estimator with lag 1 and r = 0.5:")
print("Bias:", bias_pe_l1_r05)
print("RMSE:", rmse_pe_l1_r05)
print("Coverage ratio:", coverage_pe_l1_r05)

print('----------------------')
# For Proposed estimator with lag 1 and r = 0.5
estimates_pe_l1_r3 = estimator6_df_mar.mean(axis=1).values
bias_pe_l1_r3 = bias(estimates_pe_l1_r3, true_volatility_mar)
rmse_pe_l1_r3 = RMSE(estimates_pe_l1_r3, true_volatility_mar)
coverage_pe_l1_r3, lower_ci_pe_l1_r3, upper_ci_pe_l1_r3 = ci_and_coverage_ratio(estimator6_df_mar, true_volatility_mar)

print("Proposed estimator with lag 1 and r = 3:")
print("Bias:", bias_pe_l1_r3)
print("RMSE:", rmse_pe_l1_r3)
print("Coverage ratio:", coverage_pe_l1_r3)

# For RV
estimates_rv = estimator3_df_mar.mean(axis=1).values
bias_rv = bias(estimates_rv, true_volatility_mar)
rmse_rv = RMSE(estimates_rv, true_volatility_mar)
coverage_rv, lower_ci_rv, upper_ci_rv = ci_and_coverage_ratio(estimator3_df_mar, true_volatility_mar)

print("RV estimator:")
print("Bias:", bias_rv)
print("RMSE:", rmse_rv)
print("Coverage ratio:", coverage_rv)

print('-------------------------')
# For RPV estimator with r = 2
estimates_rpv_r2 = estimator4_df_mar.mean(axis=1).values
bias_rpv_r2 = bias(estimates_rpv_r2, true_volatility_mar)
rmse_rpv_r2 = RMSE(estimates_rpv_r2, true_volatility_mar)
coverage_rpv_r2, lower_ci_rpv_r2, upper_ci_rpv_r2 = ci_and_coverage_ratio(estimator4_df_mar, true_volatility_mar)

print("RPV estimator with r = 2:")
print("Bias:", bias_rpv_r2)
print("RMSE:", rmse_rpv_r2)
print("Coverage ratio:", coverage_rpv_r2)

print('------------------------')
# For RPV estimator with r = 0.5
estimates_rpv_r05 = estimator5_df_mar.mean(axis=1).values
bias_rpv_r05 = bias(estimates_rpv_r05, true_volatility_mar)
rmse_rpv_r05 = RMSE(estimates_rpv_r05, true_volatility_mar)
coverage_rpv_r05, lower_ci_rpv_r05, upper_ci_rpv_r05 = ci_and_coverage_ratio(estimator5_df_mar, true_volatility_mar)

print("RPV estimator with r = 0.5:")
print("Bias:", bias_rpv_r05)
print("RMSE:", rmse_rpv_r05)
print("Coverage ratio:", coverage_rpv_r05)

print('------------------------')
# For RPV estimator with r = 3
estimates_rpv_r3 = estimator7_df_mar.mean(axis=1).values
bias_rpv_r3 = bias(estimates_rpv_r3, true_volatility_mar)
rmse_rpv_r3 = RMSE(estimates_rpv_r3, true_volatility_mar)
coverage_rpv_r3, lower_ci_rpv_r3, upper_ci_rpv_r3 = ci_and_coverage_ratio(estimator7_df_mar, true_volatility_mar)

print("RPV estimator with r = 3:")
print("Bias:", bias_rpv_r3)
print("RMSE:", rmse_rpv_r3)
print("Coverage ratio:", coverage_rpv_r3)
