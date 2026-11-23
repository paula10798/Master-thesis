import numpy as np
from scipy import stats


def bias(estimates, true_volatility):
    """
    Computing the bias

    Input:
    - estimates (pd.Series) = estimates for volatility
    - true_volatility (int) = value of true volatility

    Output:
    - np.mean(estimates - true_volatility) = the bias, as an integer
    """
    return np.mean(estimates - true_volatility)

def RMSE(estimates, true_volatility):
    """
    Compute the Root Mean Squared Error (RMSE)

    Input:
    - estimates (pd.Series) = estimates for volatility
    - true_volatility (int) = value of true volatility

    Output:
    - np.sqrt(np.mean((estimates - true_volatility) **2)) = Estimate for RMSE
    """
    
    return np.sqrt(np.mean((estimates - true_volatility) ** 2))

def ci_and_coverage_ratio(all_sims_df, true_volatility, alpha=0.05):
    """
    Compute the confidence intervals and coverage ratio

    Input:
    - all_sims_df (pd.DataFrame) = DataFrame containing simulated output across M simulations
    - true_volatility (int) = value of true volatility

    Output:
    - coverage (int) = Estimate for the coverage ratio
    - lower_ci (int) = lower confidence interval
    - upper_ci (int) = upper confidence interval
    """
    z = stats.norm.ppf(1 - alpha/2)
    # Mean and std across simulations at each time point
    means = all_sims_df.mean(axis=0)
    ses = all_sims_df.std(axis=0, ddof=1) / np.sqrt(all_sims_df.shape[0])
    lower_ci = means - (z * ses)
    upper_ci = means + (z * ses)
    inside = (true_volatility >= lower_ci) & (true_volatility <= upper_ci)
    coverage = inside.mean()
    return coverage, lower_ci, upper_ci
