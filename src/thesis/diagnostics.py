from statsmodels.stats.diagnostic import acorr_ljungbox


def test_autocorrelation(series, label, lags):
    """
    Computing the Ljung-Box test statistic and conclusions

    Input:
    - series (pd.DataFrame) = Data for which you want to test the presence of autocorrelation up to some lag using the LB test
    - label () = Type of returns
    - lags (int) = Number of lags

    Output:
    - It prints out the conclusion i.e. whether there is or isn't autocorrelation present up to some lag
    """
    lb_test = acorr_ljungbox(series, lags=lags)
    for idx, lag in enumerate(lags):
        stat = lb_test['lb_stat'].iloc[idx]
        pval = lb_test['lb_pvalue'].iloc[idx]
        if pval < 0.05:
            print(f"There was autocorrelation detected up to the lag of {lag} for {label} returns")
        else:
            print(f"There was no autocorrelation detected up to the lag of {lag} for {label} returns")
        print(f"Lag: {lag}, Test Statistic: {stat:.4f}, p-value: {pval:.4f}")
