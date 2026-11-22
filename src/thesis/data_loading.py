import pandas as pd
from config import file_path_nontrading_period, file_path_trading_period, sheet_names


def load_nontrading():
    dfs = {}
    for sheet in sheet_names:
        df = pd.read_excel(file_path_nontrading_period, sheet_name=sheet, engine='openpyxl')
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        dfs[sheet] = df
    return dfs


def load_trading():
    trading_df = pd.read_excel(file_path_trading_period,
                               sheet_name='Trading Period - 5min',  # Using 5-min intervals
                               engine='openpyxl')
    trading_df.iloc[:, 0] = pd.to_datetime(trading_df.iloc[:, 0])
    trading_df.set_index(trading_df.columns[0], inplace=True)
    return trading_df
