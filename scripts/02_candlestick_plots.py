import mplfinance as mpf

from thesis.data_loading import load_trading
from thesis.constants import interval_per_year_5min
from thesis.estimators.ok import OK_estimator

trading_df = load_trading()
candlestick_data = trading_df

## Candlestick Analysis ##

# Plotting candlestick charts over sample period
# Requires zooming in to see the charts, as too many data points

print('-------------------------------------------')
print('We are now plotting the candlestick charts over the 5-year sample period')

candlestick_data = candlestick_data.drop('R_tradingperiod', axis=1)

mpf.plot(candlestick_data, type='candle', style='charles',
         title='5-Min Candlesticks Plotted Over Time', ylabel='Price',
         datetime_format='%Y-%m-%d')
mpf.show()

print('-----------')
print('Since the high-low range and absolute close-to-open return is difficult to infer from the previous, we focus on particular days')
print('-----------')

print('Candlestick charts for February 3rd, 2021 — during a period of rising COVID-19 cases in Italy — are shown in the following graph:')
start_date = '2021-02-23' # Spike in Covid-19 cases in Italy
filtered_data = candlestick_data.loc[start_date]
mpf.plot(filtered_data, type='candle', style='charles',
         title='5-Min Candlesticks for 24/02/2021', ylabel='Price')
mpf.show()

print('Candlestick charts for April 9th, 2025 - during which markets reacted to the U.S. tariff announcements made the previous day - are displayed in the graph:')
start_date = '2025-04-9' # Reaction following US tariff announcements from day before
filtered_data = candlestick_data.loc[start_date]
mpf.plot(filtered_data, type='candle', style='charles',
         title='5-Min Candlesticks for 09/04/2025', ylabel='Price')
mpf.show()

print('Candlestick charts for March 15th, 2023 are presented in the following graph:')
start_date = '2023-03-15'
filtered_data = candlestick_data.loc[start_date]
mpf.plot(filtered_data, type='candle', style='charles',
         title='5-Min Candlesticks for 15/03/2023', ylabel='Price')
mpf.show()

## OK estimator according to Li et al. (2024) ##

OK_estimates = OK_estimator(trading_df['High'], trading_df['Low'],
                            trading_df['Close'], trading_df['Open'],
                            interval_per_year_5min)
