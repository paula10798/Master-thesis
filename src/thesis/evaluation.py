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

def true_volatility_calculate(params_gaussian, params_student_t, mixture_weight_gaussian):
    """
    Compute the true volatility

    Input:
    - params_gaussian (list) = Parameters used for the Gaussian distribution in the simulation
    - params_student_t (list) = Parameters used for the Student-t distribution in the simulation
    - mixture_weight_gaussian (int) = Mixture weight assigned to the Gaussian distribution, in the simulation

    Output:
    - volatility (int) = final true volatility
    """
    mu1, sigma_squared_1 = params_gaussian
    mu2, sigma_squared_2, df = params_student_t
    
    w_1 = mixture_weight_gaussian
    w_2 = 1 - w_1

    overall_mean = (w_1*mu1) + (w_2*mu2)

    component1 = w_1 * sigma_squared_1
    component2 = w_2 * ((df/(df - 2))*sigma_squared_2)
    component3 = overall_mean**2

    variance = component1 + component2 + component3
    volatility = np.sqrt(variance)

    return volatility
