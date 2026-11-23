import numpy as np
import pandas as pd


def OK_estimator(High, Low, Close, Open, interval_per_year, w_1=0.811, w_2=(-0.369)):
    """
    Compute OK estimator, according to Li et al.'s (2024) definition

    Input:
    - High (pd.DataFrame) = High prices for each 5-minute interval, in ascending time order
    - Low (pd.DataFrame) = Low prices for each 5-minute interval, in ascending time order
    - Close (pd.DataFrame) = Close prices for each 5-minute interval, in ascending time order
    - Open (pd.DataFrame) = Open prices for each 5-minute interval, in ascending time order
    - interval_per_year (int) = Number of intervals in a year

    Output:
    - OK_volatility (pd.DataFrame) = OK estimates for volatility
    """

    HL = High - Low

    absCO = abs(Close - Open)

    delta_n = 1 / interval_per_year

    OK_volatility = ((w_1 * HL) + (w_2 * absCO)) / (delta_n**0.5)
    OK_volatility = pd.DataFrame(OK_volatility)

    return OK_volatility


def OK_BV_intraday(High, Low, Close, Open, w_1, w_2, r, s, interval_per_year_5min):
    """
    Compute the (r, s)-bipower variation OK estimator, at intraday level
    Using 5-min intervals

    Input:
    - High (pd.DataFrame) = High prices for each 5-minute interval, in ascending time order
    - Low (pd.DataFrame) = Low prices for each 5-minute interval, in ascending time order
    - Close (pd.DataFrame) = Close prices for each 5-minute interval, in ascending time order
    - Open (pd.DataFrame) = Open prices for each 5-minute interval, in ascending time order
    - w_1 & w_2 (int) = optimal weights that were derived according the minimal variance condition
    - interval_per_year (int) = Number of intervals in a year
    - h (int) = Total time interval (usually 1 daily)
    - r (int) = Power on current return
    - s (int) = Power on future return

    Output:
    - OK_BV_estimates (pd.DataFrame) = (r, s)-bipower variation OK estimates for volatility
    """

    HL = High - Low
    absCO = abs(Close - Open)

    M = len(High)

    h = np.arange(1, M)
    factor = (h / M)**(1 - ((r+s)/2))

    delta_n = 1 / interval_per_year_5min

    OK_BV = np.zeros(M-1)

    for i in range(M - 1):
        OK_BV[i] = ((w_1 * (absCO.iloc[i]**r) * (absCO.iloc[i+1]**s)) + (w_2 * (HL.iloc[i]**r) * (HL.iloc[i+1]**s))) / (delta_n**0.5)

    result = factor * OK_BV
    OK_BV_estimates = pd.DataFrame(result)

    return OK_BV_estimates
