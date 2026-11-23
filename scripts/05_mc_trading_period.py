import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from thesis.constants import interval_per_year_5min
from thesis.estimators.ok import OK_estimator, OK_BV_intraday
from thesis.simulation.ohlc import compute_ohlc_for_sim
from thesis.evaluation import bias, RMSE, ci_and_coverage_ratio

## Monte Carlo Simulation for Trading Period ##


n = interval_per_year_5min * 7 # =148,176 time steps in a year for (5/7)-min frequency
T = 1 # time in years
S0 = 100 # initial price
M = 1000 # Number of simulations

dt = T/n


# QUIET PERIOD - (Geometric Brownian Motion)
print('Simulating the Quiet environment (GBM)...')
print('-----------')

u_q = 0 # drift coefficient
sigma_q = 0.15 # annual volatility
sigma_step_q = sigma_q * np.sqrt(dt)

rt_q = np.exp((mu_q - (sigma_q**2)/2) * dt + sigma_q * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)

St_q = np.vstack([np.ones(M), rt_q])
St_q = S0 * St_q.cumprod(axis=0)

time_q = np.linspace(0, T, n+1)
#tt_q = np.full(shape=(M, n+1), fill_value=time_q).T

print('Plotting the simulated GBM:')
plt.plot(time_q, St_q)
plt.xlabel("Years $(t)$")
plt.ylabel("Prices $(P_t)$")
plt.title("Realizations of Geometric Brownian Motion")
plt.show()

print('---------------')

merged = np.column_stack([time_q, St_q])
column_names = ['time'] + [f'sim_{i+1}' for i in range(M)]
simulation_quiet_tick = pd.DataFrame(merged, columns=column_names)


# Begin the rescaling from 5/7 min to 5mins

# Example usage for one simulation:
# df is your full dataframe with columns: 'time', 'sim_1', ..., 'sim_100'
sim_col = 'sim_1'
ohlc_sim1_quiet = compute_ohlc_for_sim(simulation_quiet_tick, sim_col)
#print(ohlc_sim1_quiet.head())


# To do this for all simulations:
ohlc_dict_quiet = {}
sim_cols = [col for col in simulation_quiet_tick.columns if col.startswith('sim_')]
for sim_col in sim_cols:
    ohlc_dict_quiet[sim_col] = compute_ohlc_for_sim(simulation_quiet_tick, sim_col)


#------------------------#
# Computing volatility estimates, using simulated data
#------------------------#
    
# Estimates for simple OK

OK_simple_intraday_simulated_quiet = {}
for sim, ohlc_df_quiet in ohlc_dict_quiet.items():
    vol_est = OK_estimator(ohlc_df_quiet['high'], ohlc_df_quiet['low'],
                           ohlc_df_quiet['close'], ohlc_df_quiet['open'],
                           interval_per_year_5min)
    OK_simple_intraday_simulated_quiet[sim] = vol_est

all_OK_simple_df = pd.concat(
    [df.rename(columns={df.columns[0]: sim}) for sim, df in OK_simple_intraday_simulated_quiet.items()],
    axis=1
)

average_OK_simple_quiet = all_OK_simple_df.mean(axis=1)
average_OK_simple_quiet_df = average_OK_simple_quiet.to_frame(name="average_OK_simple")


# Estimates for OK BV r=s=0.5

OK_BV_r05s05_intraday_simulated_quiet = {}
for sim, ohlc_df_quiet in ohlc_dict_quiet.items():
    vol_est = OK_BV_intraday(ohlc_df_quiet['high'], ohlc_df_quiet['low'],
                             ohlc_df_quiet['close'], ohlc_df_quiet['open'],
                             -0.163480, 0.391136,
                             0.5, 0.5, interval_per_year_5min)
    OK_BV_r05s05_intraday_simulated_quiet[sim] = vol_est

all_OK_BV_r05s05_df = pd.concat(
    [df.rename(columns={df.columns[0]: sim}) for sim, df in OK_BV_r05s05_intraday_simulated_quiet.items()],
    axis=1
)

average_OK_BV_r05s05_quiet = all_OK_BV_r05s05_df.mean(axis=1)
average_OK_BV_r05s05_quiet_df = average_OK_BV_r05s05_quiet.to_frame(name="average_OK_BV")

## For OK BV r =2 and s = 1

OK_BV_r2s1_intraday_simulated_quiet = {}
for sim, ohlc_df_quiet in ohlc_dict_quiet.items():
    vol_est = OK_BV_intraday(ohlc_df_quiet['high'], ohlc_df_quiet['low'],
                             ohlc_df_quiet['close'], ohlc_df_quiet['open'],
                             -0.032097, 0.032459,
                             2, 1, interval_per_year_5min)
    OK_BV_r2s1_intraday_simulated_quiet[sim] = vol_est

all_OK_BV_r2s1_df = pd.concat(
    [df.rename(columns={df.columns[0]: sim}) for sim, df in OK_BV_r2s1_intraday_simulated_quiet.items()],
    axis=1
)

average_OK_BV_r2s1_quiet = all_OK_BV_r2s1_df.mean(axis=1)
average_OK_BV_r2s1_quiet_df = average_OK_BV_r2s1_quiet.to_frame(name="average_OK_BV")

## For OK BV r = 3 and s = 1

OK_BV_r3s1_intraday_simulated_quiet = {}
for sim, ohlc_df_quiet in ohlc_dict_quiet.items():
    vol_est = OK_BV_intraday(ohlc_df_quiet['high'], ohlc_df_quiet['low'],
                             ohlc_df_quiet['close'], ohlc_df_quiet['open'],
                             -0.006871, 0.007464,
                             3, 1, interval_per_year_5min)
    OK_BV_r3s1_intraday_simulated_quiet[sim] = vol_est

all_OK_BV_r3s1_df = pd.concat(
    [df.rename(columns={df.columns[0]: sim}) for sim, df in OK_BV_r3s1_intraday_simulated_quiet.items()],
    axis=1
)

average_OK_BV_r3s1_quiet = all_OK_BV_r3s1_df.mean(axis=1)
average_OK_BV_r3s1_quiet_df = average_OK_BV_r3s1_quiet.to_frame(name="average_OK_BV")


## For OK BV r = 1 and s = 3

OK_BV_r1s3_intraday_simulated_quiet = {}
for sim, ohlc_df_quiet in ohlc_dict_quiet.items():
    vol_est = OK_BV_intraday(ohlc_df_quiet['high'], ohlc_df_quiet['low'],
                             ohlc_df_quiet['close'], ohlc_df_quiet['open'],
                             -0.010443, 0.008298,
                             1, 3, interval_per_year_5min)
    OK_BV_r1s3_intraday_simulated_quiet[sim] = vol_est

all_OK_BV_r1s3_df = pd.concat(
    [df.rename(columns={df.columns[0]: sim}) for sim, df in OK_BV_r1s3_intraday_simulated_quiet.items()],
    axis=1
)

average_OK_BV_r1s3_quiet = all_OK_BV_r1s3_df.mean(axis=1)
average_OK_BV_r1s3_quiet_df = average_OK_BV_r1s3_quiet.to_frame(name="average_OK_BV")

#-------------------------#
# Evaluating the performance of estimators
#-------------------------#

true_volatility = sigma_q

print('---------------------')
# For OK_simple
estimates_simple = average_OK_simple_quiet_df['average_OK_simple'].values
bias_simple = bias(estimates_simple, true_volatility)
rmse_simple = RMSE(estimates_simple, true_volatility)
coverage_simple, lower_ci_simple, upper_ci_simple = ci_and_coverage_ratio(all_OK_simple_df, true_volatility)

print("OK_simple estimator:")
print("Bias:", bias_simple)
print("RMSE:", rmse_simple)
print("Coverage ratio:", coverage_simple)

print('----------------------')
# For OK BV r=s=0.5
estimates_bv_r05s05 = average_OK_BV_r05s05_quiet_df['average_OK_BV'].values
bias_bv_r05s05 = bias(estimates_bv_r05s05, true_volatility)
rmse_bv_r05s05 = RMSE(estimates_bv_r05s05, true_volatility)
coverage_bv_r05s05, lower_ci_bv_r05s05, upper_ci_bv_r05s05 = ci_and_coverage_ratio(all_OK_BV_r05s05_df, true_volatility)

print("OK_BV r05s05 estimator:")
print("Bias:", bias_bv_r05s05)
print("RMSE:", rmse_bv_r05s05)
print("Coverage ratio:", coverage_bv_r05s05)

print('---------------------')
# For OK BV r=2 and s=1
estimates_bv_r2s1 = average_OK_BV_r2s1_quiet_df['average_OK_BV'].values
bias_bv_r2s1 = bias(estimates_bv_r2s1, true_volatility)
rmse_bv_r2s1 = RMSE(estimates_bv_r2s1, true_volatility)
coverage_bv_r2s1, lower_ci_bv_r2s1, upper_ci_bv_r2s1 = ci_and_coverage_ratio(all_OK_BV_r2s1_df, true_volatility)

print('---------------------')
print("OK_BV r2s1 estimator:")
print("Bias:", bias_bv_r2s1)
print("RMSE:", rmse_bv_r2s1)
print("Coverage ratio:", coverage_bv_r2s1)

print('---------------------')
# For OK BV r=3 and s=1
estimates_bv_r3s1 = average_OK_BV_r3s1_quiet_df['average_OK_BV'].values
bias_bv_r3s1 = bias(estimates_bv_r3s1, true_volatility)
rmse_bv_r3s1 = RMSE(estimates_bv_r3s1, true_volatility)
coverage_bv_r3s1, lower_ci_bv_r3s1, upper_ci_bv_r3s1 = ci_and_coverage_ratio(all_OK_BV_r3s1_df, true_volatility)

print("OK_BV r3s1 estimator:")
print("Bias:", bias_bv_r3s1)
print("RMSE:", rmse_bv_r3s1)
print("Coverage ratio:", coverage_bv_r3s1)

print('---------------------')
# For OK BV r=1 and s=3
estimates_bv_r1s3 = average_OK_BV_r1s3_quiet_df['average_OK_BV'].values
bias_bv_r1s3 = bias(estimates_bv_r1s3, true_volatility)
rmse_bv_r1s3 = RMSE(estimates_bv_r1s3, true_volatility)
coverage_bv_r1s3, lower_ci_bv_r1s3, upper_ci_bv_r1s3 = ci_and_coverage_ratio(all_OK_BV_r1s3_df, true_volatility)

print("OK_BV r1s3 estimator:")
print("Bias:", bias_bv_r1s3)
print("RMSE:", rmse_bv_r1s3)
print("Coverage ratio:", coverage_bv_r1s3)

print('------------------------------')
