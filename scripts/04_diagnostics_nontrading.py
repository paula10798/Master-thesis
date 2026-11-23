import matplotlib.pyplot as plt

from thesis.data_loading import load_nontrading
from thesis.diagnostics import test_autocorrelation

dfs = load_nontrading()
overnight_df = dfs['Weeknight data']
weekend_df = dfs['Weekend data']

## Conducting further tests on Weekend and Overnight returns ##

# Test for normality
from scipy.stats import shapiro

normality_test_weekend = shapiro(weekend_df['R_weekend'])
normality_test_overnight = shapiro(overnight_df['R_overnight'])

print('------------------------------------------')
print("Results from the Shapiro-Wilk test:")

if normality_test_weekend[1] < 0.05:
    print("There is sufficient evidence to argue that the distribution of Weekend returns does not follow a Gaussian")
else:
    print("The distribution for Weekend returns follows a Gaussian")

if normality_test_overnight[1] < 0.05:
    print("There is sufficient evidence to argue that the distribution of Overnight returns does not follow a Gaussian")
else:
    print("The distribution for Overnight returns follows a Gaussian")

print("----------------------------------------")

from scipy.stats import kstest

ks_stat_weekend, p_value_weekend = kstest(weekend_df['R_weekend'], 'norm')
ks_stat_overnight, p_value_overnight = kstest(overnight_df['R_overnight'], 'norm')

print("Results from the Kolmogorov-Smirnov test:")

if p_value_weekend < 0.05:
    print("There is sufficient evidence to argue that the distribution of Weekend returns does not follow a Gaussian")
else:
    print("The distribution for Weekend returns follows a Gaussian")

if p_value_overnight < 0.05:
    print("There is sufficient evidence to argue that the distribution of Overnight returns does not follow a Gaussian")
else:
    print("The distribution for Overnight returns follows a Gaussian")


print("-------------------------------------------")

# Test for autocorrelation
from statsmodels.graphics.tsaplots import plot_acf

print('------------------------------------------')
print('We are now plotting the ACF plots for Overnight and Weekend log-returns')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(overnight_df['R_overnight'], lags=20, ax=axes[0])
axes[0].set_title('Autocorrelation Function Plot for Overnight returns')
plot_acf(weekend_df['R_weekend'], lags=20, ax=axes[1])
axes[1].set_title('Autocorrelation Function Plot for Weekend returns')

plt.tight_layout()
plt.show()

print('------------------------------------------')

print('Results from the LB test for Weekend log_returns')
test_autocorrelation(weekend_df['R_weekend'], "Weekend", [5])
print('------')
print('Results from the LB test for Overnight log-returns')
test_autocorrelation(overnight_df['R_overnight'], "Overnight", [1, 2, 3, 4, 5])

print('---------------------------------------------------------')
