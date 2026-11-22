import statistics as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis


def summary_stats(returns):
    """
    Compute summary statistics

    Input:
    - returns (pd.DataFrame) = Data on returns, which is stored in a column

    Output:
    - pd.DataFrame(summary_statistics) = Summary statistics are printed and stored in a Data Frame.
    """

    Mean = stats.mean(returns)
    standard_deviation = stats.stdev(returns)
    median = stats.median(returns)
    minimum = min(returns)
    maximum = max(returns)
    obs = returns.count()
    Range = maximum - minimum
    skewness = skew(returns)
    kurtosiss = kurtosis(returns)

    summary_statistics = {
        "Obs": [obs],
        "Mean": [Mean],
        "Standard Deviation": [standard_deviation],
        "Median": [median],
        "Minimum": [minimum],
        "Maximum": [maximum],
        "Range": [Range],
        "Skewness": [skewness],
        "Kurtosis": [kurtosiss]
    }

    return pd.DataFrame(summary_statistics)


def plot_returns_over_time(overnight_df, weekend_df, longweekend_df, holiday_df, trading_df):
    # For non-trading periods
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(overnight_df.index, overnight_df['R_overnight'], color="blue")
    ax1.set_xlabel("Date")
    ax1.axhline(y=0, color='y', linestyle='-')
    ax1.set_title("Overnight Log-Returns over time")

    ax2.plot(weekend_df.index, weekend_df['R_weekend'], color="red")
    ax2.set_xlabel("Date")
    ax2.axhline(y=0, color='y', linestyle='-')
    ax2.set_title("Weekend Log-Returns over time")

    longweekend_df = longweekend_df.sort_index(ascending=True)

    ax3.bar(longweekend_df.index.astype(str), longweekend_df['R_longwknd'], color="green")
    ax3.set_xlabel("Date")
    ax3.tick_params(axis='x', rotation=90, labelsize=8)
    ax3.axhline(y=0, color='y', linestyle='-')
    ax3.set_title("Long Weekend Log-Returns over time")

    holiday_df = holiday_df.sort_index(ascending=True)

    ax4.bar(holiday_df.index.astype(str), holiday_df['R_holiday'], color="purple")
    ax4.set_xlabel("Date")
    ax4.tick_params(axis='x', rotation=90, labelsize=8)
    ax4.axhline(y=0, color='y', linestyle='-')
    ax4.set_title("Holiday Log-Returns over time")

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.18, wspace=0.22, hspace=0.4)
    plt.show()

    # For trading period
    plt.plot(trading_df.index, trading_df['R_tradingperiod'], color="orange")
    plt.xlabel("Date")
    plt.axhline(y=0, color='y', linestyle='-')
    plt.title("5-Min Log-Returns During Market Opening")

    plt.show()


def plot_return_distributions(overnight_df, weekend_df, longweekend_df, holiday_df, trading_df):
    # For non-trading periods
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.distplot(overnight_df['R_overnight'], bins=1000, color='purple')
    plt.title('Overnight Log-Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')

    plt.subplot(2, 2, 2)
    sns.distplot(weekend_df['R_weekend'], bins=700, color='green')
    plt.title('Weekend Log-Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')

    plt.subplot(2, 2, 3)
    sns.distplot(longweekend_df['R_longwknd'], bins=10, color='yellow')
    plt.title('Long Weekend Log-Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')

    plt.subplot(2, 2, 4)
    sns.distplot(holiday_df['R_holiday'], bins=5, color='blue')
    plt.title('Holiday Log-Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # For trading period
    plt.figure(figsize=(10, 6))
    sns.distplot(trading_df['R_tradingperiod'], bins=400, color='orange')
    plt.title('Trading Period Log-Returns Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')

    plt.show()

    plt.figure(figsize=(10, 6))  # With a restricted range
    sns.distplot(trading_df['R_tradingperiod'], bins=500, color='orange')
    plt.title('Trading Period Log-Returns Distribution with restricted range (-0.005, 0.005)')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.xlim(-0.005, 0.005)

    plt.show()
