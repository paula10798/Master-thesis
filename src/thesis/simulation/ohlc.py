
import pandas as pd


def compute_ohlc_for_sim(df, sim_col, interval=7):
    """
    Compute OHLC for 5-minute intervals, given that the current frequency simulated earlier is (5/7)-minutes

    Input:
    - df (pd.DataFrame) = Simulated data which has a frequency of (5/7)-minutes and has M number of columns for M number of simulations
    - sim_col () = Refers to the name of the column in the DataFrame df that contains the simulated price path.

    Output:
    ohlc_df (pd.DataFrame) = Returns a DataFrame with columns that include time, open, high, low and close.
    """
    
    n_intervals = len(df) // interval
    
    vals = df[sim_col].values[:n_intervals*interval].reshape(-1, interval)
    times = df['time'].values[interval-1:n_intervals*interval:interval]
    
    ohlc_df = pd.DataFrame({
        'time': times,
        'open': vals[:, 0],
        'high': vals.max(axis=1),
        'low': vals.min(axis=1),
        'close': vals[:, -1],
    })
    
    return ohlc_df
