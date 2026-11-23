import numpy as np


def proposed_estimator(returns, s, w):
    """
    Compute volatility estimates according the proposed formula in the paper
    Only one autocovariance term is included

    Input:
    - returns (np.DataFrame) = Log-returns times series
    - s (int) =
    - w (int) = Weights for the autocovariance term

    Output:
    - sigma2 (np.DataFrame) = Volatility estimates
    """

    r = np.asarray(returns)
    p = len(w)
    sigma2 = np.full_like(r, np.nan, dtype=float)

    for t in range(p, len(r)):
        sum_autocovariances = sum(
            w[l] * r[t] * r[t - (l+1)] for l in range(p))
        sigma2[t] = (np.abs(r[t])**s) + sum_autocovariances

    return sigma2

def RV_estimator(returns):
    """
    Computes the RV estimates

    Input:
    - returns (pd.DataFrame) = Log-return series

    Output:
    - returns**2 (pd.DataFrame) = Volatility estimates according to RV formula
    """

    return returns**2

def RPV_estimator(returns, r):
    """
    Computes RPV estimates

    Input:
    - returns (pd.DataFrame) = Log-return series
    - r (int) = Power with which the absolute log-returns are being raised to

    Output:
    - np.abs(returns)**r = Volatility estimates according to RPV formula
    """

    return np.abs(returns)**r
