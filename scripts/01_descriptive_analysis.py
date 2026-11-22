from thesis.data_loading import load_nontrading, load_trading
from thesis.descriptive import (summary_stats,
                                plot_returns_over_time,
                                plot_return_distributions)

## Importing the data ##
dfs = load_nontrading()

## Accessing the dataframes ##
overnight_df = dfs['Weeknight data']
weekend_df = dfs['Weekend data']
longweekend_df = dfs['Long Weekend data']
holiday_df = dfs['Holiday data']

trading_df = load_trading()

print(overnight_df)
print(weekend_df)
print(longweekend_df)
print(holiday_df)
print(trading_df)

## Extra Preparation ##

copy = trading_df
copy['date'] = copy.index.date
unique_dates = copy['date'].unique() # Stores the dates in a separate variable

## Description of the data ##

print('---------------------------------------------------')
overnight_ss = summary_stats(overnight_df['R_overnight'])
print('Summary statistics for Overnight log-returns:')
print(overnight_ss)

print('---------------------------------------------------')
weekend_ss = summary_stats(weekend_df['R_weekend'])
print('Summary statistics for Weekend log-returns:')
print(weekend_ss)

print('---------------------------------------------------')
longweekend_ss = summary_stats(longweekend_df['R_longwknd'])
print('Summary statistics for Long Weekend log-returns:')
print(longweekend_ss)

print('---------------------------------------------------')
holiday_ss = summary_stats(holiday_df['R_holiday'])
print('Summary statistics for Holiday log-returns:')
print(holiday_ss)

print('---------------------------------------------------')
trading_ss = summary_stats(trading_df['R_tradingperiod'])
print('Summary statistics for Intraday log-returns:')
print(trading_ss)
print('---------------------------------------------------')

print('---------------------------------------------------')
print('We are now plotting the log-returns for each type of interval over time')
plot_returns_over_time(overnight_df, weekend_df, longweekend_df, holiday_df, trading_df)
print('-------------------------------------')

print('-------------------------------------')
print('We are now plotting the distribution of log-returns for each type of interval')
plot_return_distributions(overnight_df, weekend_df, longweekend_df, holiday_df, trading_df)
print('-------------------------------------')
